import React, { useEffect, useMemo, useRef, useState } from 'react'

const URL_RE = /^(https?|git|ssh):\/\/.+/i
const SSH_RE = /^git@.+:.+\.git$/i
const DONE = new Set(['succeeded', 'failed', 'cancelled'])
const validUrl = (v) => URL_RE.test(v.trim()) || SSH_RE.test(v.trim())
const fmtUsd = (v) => (v === null || v === undefined ? '-' : `$${Number(v).toFixed(6)}`)
const readLastUrl = () => {
  try {
    return localStorage.getItem('sutra:lastRepoUrl') || ''
  } catch {
    return ''
  }
}
const writeLastUrl = (value) => {
  try {
    localStorage.setItem('sutra:lastRepoUrl', value)
  } catch {}
}
const dur = (s, f) => {
  if (!s || !f) return '-'
  const ms = new Date(f).getTime() - new Date(s).getTime()
  return Number.isFinite(ms) && ms >= 0 ? `${(ms / 1000).toFixed(1)}s` : '-'
}

export function App() {
  const [repoUrl, setRepoUrl] = useState(readLastUrl)
  const [replace, setReplace] = useState(true)
  const [error, setError] = useState('')
  const [activeJobId, setActiveJobId] = useState('')
  const [activeJob, setActiveJob] = useState(null)
  const [history, setHistory] = useState([])
  const [logLines, setLogLines] = useState([])
  const logRef = useRef(null)
  const eventSrcRef = useRef(null)

  const isActive = activeJob && (activeJob.status === 'queued' || activeJob.status === 'running')
  const canSubmit = useMemo(() => validUrl(repoUrl), [repoUrl])
  const summary = activeJob?.summary || null

  const fetchHistory = async () => {
    const res = await fetch('/api/jobs')
    const data = await res.json()
    setHistory(data)
    if (!activeJobId && data.length) setActiveJob(data[0])
  }

  useEffect(() => {
    void fetchHistory()
    return () => eventSrcRef.current?.close()
  }, [])

  useEffect(() => {
    if (!activeJobId) return
    eventSrcRef.current?.close()
    const es = new EventSource(`/api/jobs/${activeJobId}/stream`)
    eventSrcRef.current = es

    es.addEventListener('log', (ev) => {
      const p = JSON.parse(ev.data)
      setLogLines((prev) => [...prev, `[${p.source}] ${p.line}`].slice(-2000))
    })
    es.addEventListener('started', (ev) => setActiveJob(JSON.parse(ev.data)))
    es.addEventListener('completed', (ev) => {
      setActiveJob(JSON.parse(ev.data))
      void fetchHistory()
      es.close()
      eventSrcRef.current = null
    })
    es.onerror = () => {
      es.close()
      eventSrcRef.current = null
    }
    return () => es.close()
  }, [activeJobId])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logLines])

  const onSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!validUrl(repoUrl)) return setError('Please enter a valid git URL')
    writeLastUrl(repoUrl.trim())
    setLogLines([])
    const res = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repoUrl: repoUrl.trim(), replace }),
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: 'Failed to submit job' }))
      return setError(body.detail || 'Failed to submit job')
    }
    const data = await res.json()
    setActiveJobId(data.jobId)
    const job = await (await fetch(`/api/jobs/${data.jobId}`)).json()
    setActiveJob(job)
    void fetchHistory()
  }

  const onCancel = async () => activeJobId && fetch(`/api/jobs/${activeJobId}/cancel`, { method: 'POST' })
  const onSelectHistory = async (jobId) => {
    setActiveJobId(jobId)
    setLogLines([])
    setActiveJob(await (await fetch(`/api/jobs/${jobId}`)).json())
  }

  return (
    <main className="page">
      <h1>Sutra Local UI</h1>
      <section className="card">
        <h2>Submit Job</h2>
        <form onSubmit={onSubmit}>
          <textarea className="url-input" rows={4} placeholder="https://github.com/gin-gonic/gin" value={repoUrl} onChange={(e) => setRepoUrl(e.target.value)} />
          <label className="check-row"><input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />Replace existing index</label>
          <button type="submit" disabled={!canSubmit}>Start Indexing</button>
        </form>
        {error && <p className="error">{error}</p>}
      </section>

      {isActive && (
        <section className="card">
          <h2>Active Job</h2>
          <p><strong>Repo:</strong> {activeJob.repoUrl}</p>
          <p><strong>Status:</strong> {activeJob.status}</p>
          <p><strong>Queue Position:</strong> {activeJob.queuePosition ?? '-'}</p>
          <div className="log-panel" ref={logRef}><pre>{logLines.join('\n')}</pre></div>
          <button onClick={onCancel}>Cancel</button>
        </section>
      )}

      {activeJob && DONE.has(activeJob.status) && (
        <section className="card">
          <h2>Results</h2>
          <p><strong>Status:</strong> {activeJob.status}</p>
          <p><strong>Exit Code:</strong> {activeJob.exitCode ?? '-'}</p>
          <p><strong>Duration:</strong> {dur(activeJob.startedAt, activeJob.finishedAt)}</p>
          {summary && <>
            <p><strong>Symbols:</strong> {summary.symbol_count ?? '-'}</p>
            <p><strong>Files:</strong> {summary.file_count ?? '-'}</p>
            <p><strong>Commit:</strong> {summary.commit_sha ?? '-'}</p>
            <p><strong>Languages:</strong> {JSON.stringify(summary.languages || {})}</p>
            <p><strong>Embedding Model:</strong> {summary.embedding_model ?? '-'}</p>
            <p><strong>Embedding Tokens:</strong> {summary.embedding_total_tokens ?? '-'}</p>
            <p><strong>Estimated API Cost:</strong> {fmtUsd(summary.embedding_estimated_cost_usd)}</p>
          </>}
          <p><a href={`/api/jobs/${activeJob.id}/artifacts/graph.json`}>graph.json</a></p>
          <p><a href={`/api/jobs/${activeJob.id}/artifacts/embeddings.npy`}>embeddings.npy</a></p>
          <p><a href={`/api/jobs/${activeJob.id}/artifacts/embeddings_index.json`}>embeddings_index.json</a></p>
          {activeJob.status === 'failed' && <>
            <p className="error">{activeJob.error || 'Job failed'}</p>
            {activeJob.errorDetail && <p className="error"><strong>Details:</strong> {activeJob.errorDetail}</p>}
            <div className="log-tail"><pre>{logLines.slice(-50).join('\n')}</pre></div>
          </>}
        </section>
      )}

      <section className="card">
        <h2>History (Last 50)</h2>
        <div className="history-list">
          {history.map((job) => (
            <button key={job.id} className="history-row" onClick={() => onSelectHistory(job.id)}>
              <span>{job.repoUrl}</span>
              <span>{job.createdAt}</span>
              <span>{job.status}</span>
              <span>{job.summary?.symbol_count ?? '-'}</span>
              <span>{fmtUsd(job.summary?.embedding_estimated_cost_usd ?? job.embeddingEstimatedCostUsd)}</span>
            </button>
          ))}
        </div>
      </section>
    </main>
  )
}
