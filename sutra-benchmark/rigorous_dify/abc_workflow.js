export const meta = {
  name: 'dify-single-abc-benchmark',
  description: 'Single-repo (dify) A/B/C (SUTRA_ONLY/GREP_ONLY/BOTH) code-tracing benchmark: solve + blind-grade each cell, n=6 trials, re-run at d67123e with time metric',
  phases: [{ title: 'Solve' }, { title: 'Grade' }],
}

// ---- config ----
const REPOS = {
  dify: { dir: '/tmp/claude-1000/-home-ritik-Desktop-sutra/5d5da943-8e2c-467b-8fbe-07a9b7f411aa/scratchpad/dify_src',
          sutra_repo: 'langgenius/dify',
          desc: 'the Dify codebase (large open-source LLMOps platform; Python backend under api/)' },
}

const SOLVE_MODEL = 'claude-sonnet-5'
const GRADE_MODEL = 'claude-sonnet-5'

// exact tickets from the 2-repo run (d67123e), reused verbatim for comparability
const TICKETS = {
  dify: [
    { id:'DL1', text:'Trace how retrieval reranks candidate documents and applies the top-k cutoff, and where results below the score threshold are dropped.' },
    { id:'DL2', text:'Trace how Dify checks a workspace’s billing quota before running a model call, and where an over-quota call is rejected.' },
    { id:'DL3', text:'Trace how a workflow variable reference like {{#node.field#}} is parsed and resolved to a concrete value, and where an unresolved reference fails.' },
    { id:'DS1', text:'Where does Dify stop a single application from taking all the in-flight request slots so other applications still get served? Trace where the count is checked and where it is released.' },
    { id:'DS2', text:'A generation that keeps going far too long should be cut off rather than run forever. Trace where Dify force-stops such a run and what event it emits.' },
    { id:'DS3', text:'A document that repeatedly fails to index should not be retried endlessly and should end up visibly failed. Trace where that failure is recorded and the retry stops.' },
  ],
}

// gold "must contain" facts for blind grading (verbatim from the 2-repo run, re-verified at d67123e)
const GOLD = {
  'dify|DL1': ['rerank in api/core/rag/rerank/rerank_model.py RerankModelRunner.run','score_threshold filter drops below-threshold results','top_n / top-k truncation'],
  'dify|DL2': ['QuotaService reserve/consume before the model call (api/services/quota_service.py)','QuotaExceededError raised when over quota'],
  'dify|DL3': ['{{#node.field#}} resolved by VariablePool.convert_template','the resolver lives in the external graphon package (not in the repo)','unresolved-ref behavior (kept raw / ValueError)'],
  'dify|DS1': ['per-app active-request cap: RateLimit.enter checks Redis hlen vs max_active_requests (api/core/app/features/rate_limiting/rate_limit.py)','exit/close decrements the counter'],
  'dify|DS2': ['AppQueueManager.listen: elapsed > APP_MAX_EXECUTION_TIME force-publishes QueueStopEvent (api/core/app/apps/base_app_queue_manager.py)','the stop is emitted as a QueueStopEvent that ends the run'],
  'dify|DS3': ['IndexingRunner._handle_indexing_error sets document indexing_status = error (api/core/indexing_runner.py ~57)','retries are user-triggered/guarded (retry_document_indexing_task), not endless auto-retry'],
}

const ARM_TOOLS = {
  SUTRA_ONLY: 'TOOLS ALLOWED: ONLY the Sutra MCP tools and the Read tool. First load the Sutra tools with ONE call: ToolSearch query "select:mcp__sutra__sutra_search,mcp__sutra__sutra_get_symbol,mcp__sutra__sutra_get_callers,mcp__sutra__sutra_get_callees,mcp__sutra__sutra_expand_neighbors". Navigate EXCLUSIVELY via those sutra_* tools (always pass repo="{SUTRA_REPO}"); use Read only to confirm a specific file:line. You are STRICTLY FORBIDDEN from using Bash, Grep, or Glob — do not call them for any reason.',
  GREP_ONLY:  'TOOLS ALLOWED: ONLY Bash, Grep, Glob, and Read. You are STRICTLY FORBIDDEN from using or loading ANY tool whose name contains "sutra" — treat them as nonexistent; never call ToolSearch to load them.',
  BOTH:       'TOOLS ALLOWED: any of — the Sutra MCP tools (load via ToolSearch "select:mcp__sutra__sutra_search,mcp__sutra__sutra_get_symbol,mcp__sutra__sutra_get_callers,mcp__sutra__sutra_get_callees,mcp__sutra__sutra_expand_neighbors"; repo="{SUTRA_REPO}"), Bash, Grep, Glob, and Read. Use whichever you prefer.',
}

const ANSWER_SCHEMA = { type:'object', additionalProperties:false,
  properties:{ answer:{ type:'string', description:'A concise root-cause trace answering the ticket, with concrete file:line references. Do NOT mention which tools you used.' } }, required:['answer'] }
const GRADE_SCHEMA = { type:'object', additionalProperties:false,
  properties:{ score:{ type:'integer', minimum:0, maximum:4 }, justification:{ type:'string', description:'one sentence' } }, required:['score','justification'] }

function solvePrompt(cell){
  const r = REPOS[cell.repo]
  const armTools = ARM_TOOLS[cell.arm].replace(/\{SUTRA_REPO\}/g, r.sutra_repo)
  return `You are a senior engineer investigating ${r.desc}. The repo is checked out at the exact indexed commit at: ${r.dir}. Work ONLY within that directory. (Never access any other copy of the repo on disk.)

${armTools}

TICKET: ${cell.ticket}

Investigate and produce a concise root-cause / trace answer backed by concrete file:line references. Be efficient. Return ONLY the answer via the structured output; do not describe your process or which tools you used.`
}

function gradePrompt(cell, candidate){
  const facts = (GOLD[`${cell.repo}|${cell.tid}`]||[]).map((f,i)=>`  ${i+1}. ${f}`).join('\n')
  return `You are blind-grading one candidate answer to a code-tracing ticket against ground truth. You do NOT know how the candidate was produced; judge ONLY correctness versus the ground truth.

TICKET: ${cell.ticket}

GROUND TRUTH — a correct answer must identify these facts:
${facts}

CANDIDATE ANSWER:
"""
${candidate}
"""

Score 0-4:
 0 = wrong or fabricated (names wrong files/mechanism, or invents things)
 1 = partially right but wrong root cause
 2 = right area, but mechanism vague or missing the key fact(s)
 3 = correct: right file(s):line and the correct mechanism (covers the ground-truth facts)
 4 = correct AND surfaces a real, accurate edge case or bug beyond the ground truth
Return score + a one-sentence justification.`
}

// ---- build cells: 6 tickets x 3 arms x 6 trials = 108 ----
const ARMS = ['SUTRA_ONLY','GREP_ONLY','BOTH']
const TRIALS = [1,2,3,4,5,6]
let CELLS = []
for (const repo of Object.keys(TICKETS))
  for (const t of TICKETS[repo])
    for (const arm of ARMS)
      for (const trial of TRIALS)
        CELLS.push({ id:`${repo}|${t.id}|${arm}|t${trial}`, repo, tid:t.id, arm, trial, ticket:t.text })

let A = (typeof args !== 'undefined' && args) ? args : {}
if (typeof A === 'string') { try { A = JSON.parse(A) } catch { A = {} } }
if (A.arms)  CELLS = CELLS.filter(c => A.arms.includes(c.arm))
if (A.trials) CELLS = CELLS.filter(c => A.trials.includes(c.trial))
if (A.tickets) CELLS = CELLS.filter(c => A.tickets.includes(c.tid))
if (A.only)  CELLS = CELLS.filter(c => A.only.includes(c.id))
if (A.limit) CELLS = CELLS.slice(0, A.limit)
const MODE = A.mode || 'both'

log(`cells=${CELLS.length} mode=${MODE} solve_model=${SOLVE_MODEL}`)

const results = await pipeline(
  CELLS,
  (c) => agent(solvePrompt(c), { label:`solve|${c.id}`, phase:'Solve', schema:ANSWER_SCHEMA, model:SOLVE_MODEL, effort:'medium' })
           .then(r => ({ cell:c.id, answer: r ? r.answer : null })),
  (solveRes, c) => {
    if (MODE === 'solve' || !solveRes || !solveRes.answer) return { cell:c.id, answer: solveRes ? solveRes.answer : null, score:null }
    return agent(gradePrompt(c, solveRes.answer), { label:`grade|${c.id}`, phase:'Grade', schema:GRADE_SCHEMA, model:GRADE_MODEL, effort:'low' })
             .then(g => ({ cell:c.id, answer:solveRes.answer, score: g ? g.score : null, justification: g ? g.justification : null }))
  }
)

return results.filter(Boolean)
