export const meta = {
  name: 'sutra-gold-scout',
  description: 'Author verified ground-truth gold for candidate sutra-repo tracing tickets (overseer mode: grep + sutra cross-checked)',
  phases: [{ title: 'Scout' }],
}

const WT = '/tmp/claude-1000/-home-ritik-Desktop-sutra/5d5da943-8e2c-467b-8fbe-07a9b7f411aa/scratchpad/sutra_src'

const GOLD_SCHEMA = { type:'object', additionalProperties:false, properties:{
  slug:{type:'string'},
  class_hint:{type:'string', enum:['lexical','semantic']},
  ticket_symptom:{type:'string', description:'a symptom-first phrasing a user would file, describing the observable behavior WITHOUT naming the internal mechanism/function/keyword'},
  entry_points:{type:'array', items:{type:'string'}, description:'file:line anchors, each VERIFIED to exist at this commit (state the symbol name too)'},
  mechanism:{type:'string', description:'2-4 sentence precise mechanism, the real root-cause trace'},
  must_contain:{type:'array', items:{type:'string'}, description:'2-3 ATOMIC facts a correct answer MUST state (each independently checkable)'},
  naive_lexical_keyword:{type:'string', description:'the single most obvious word a dev would grep from the symptom'},
  naive_lexical_count:{type:'integer', description:'grep -rIiw --include=*.py count of that word under sutra/'},
  lands_near_answer:{type:'boolean', description:'does grepping naive_lexical_keyword land the dev at/near the gold file? (true => lexical ticket)'},
  symptom_keywords_absent:{type:'array', items:{type:'string'}, description:'symptom words (from the symptom framing) that are ~absent in code (0 or near-0 count)'},
  answer_fully_in_repo:{type:'boolean', description:'true iff EVERY gold fact lives inside this repo (NOT an external pip package)'},
  ambiguity_risk:{type:'string', description:'is there MORE THAN ONE valid mechanism/location a reasonable engineer could land on? name them, or say "none - single unambiguous mechanism"'},
  confidence:{type:'string', enum:['high','medium','low']}
}, required:['slug','class_hint','ticket_symptom','entry_points','mechanism','must_contain','naive_lexical_keyword','naive_lexical_count','lands_near_answer','symptom_keywords_absent','answer_fully_in_repo','ambiguity_risk','confidence'] }

function scoutPrompt(c){
  return `You are the OVERSEER authoring ground-truth for a code-tracing benchmark on the 'sutra' codebase (a code-index/retrieval engine: extractor -> embedder -> graph/vector store -> MCP retrieval). The repo is checked out at the exact indexed commit at: ${WT}. Work ONLY within that directory.

You have BOTH toolsets and should CROSS-CHECK with both:
 - grep/Bash/Glob/Read over ${WT}
 - the Sutra index: load with ONE ToolSearch "select:mcp__sutra__sutra_search,mcp__sutra__sutra_get_symbol,mcp__sutra__sutra_get_callers,mcp__sutra__sutra_get_callees,mcp__sutra__sutra_expand_neighbors" and always pass repo="sutra".

CANDIDATE MECHANISM TO PIN DOWN:
${c.desc}

Your job: produce VERIFIED ground truth for a ticket about this mechanism.
 1. Find the REAL entry point(s) and the exact file:line where the mechanism happens. Open the file and confirm the line numbers are correct at THIS commit (do not trust memory).
 2. Write the precise mechanism (the actual root-cause trace).
 3. Write 2-3 atomic must_contain facts a correct answer MUST state — each must be independently checkable and UNAMBIGUOUS.
 4. Determine the naive lexical keyword a dev greps from the SYMPTOM, run 'grep -rIiw --include=*.py <kw> sutra | wc -l' to get its count, and judge whether that grep lands them near the gold file (lands_near_answer).
 5. From the symptom framing, list 1-3 symptom words and verify (by grep) they are ~absent (0/near-0). These make it a SEMANTIC ticket.
 6. CRITICAL: verify answer_fully_in_repo — every gold fact must live in THIS repo, not in an external installed package. If the core of the mechanism is in a third-party dependency, say answer_fully_in_repo=false and explain.
 7. CRITICAL: assess ambiguity_risk honestly — if two different files/mechanisms could each be a "correct" answer, say so (we will DROP ambiguous tickets, like a prior run's dify DS2).

Be rigorous and skeptical. Return ONLY the structured gold.`
}

const CANDIDATES = [
  { slug:'rerank_cutoff', desc:'How sutra re-scores the initial search candidate set with a heavier second-pass (cross-encoder) model and applies the final top-k cutoff, dropping candidates below the cutoff. Look at sutra/core/retrieval/reranker.py and how the retrieval pipeline invokes it (rerank flag) and truncates.' },
  { slug:'moniker_build', desc:'How sutra builds the stable identifier ("moniker") used to refer to a code symbol across re-indexes, and what disambiguates two symbols that share the same name (e.g. overloads / nested scope / file path). Look at sutra/core/extractor/moniker.py.' },
  { slug:'incremental_changed_files', desc:'How sutra re-indexes only the files that changed instead of the whole repo: how it computes the changed-file set (git diff between the old indexed commit and the new one) and how it treats added vs modified vs deleted files. Look at sutra/core/git_differ.py and sutra/core/incremental_updater.py.' },
  { slug:'repo_isolation_read', desc:'A single sutra server serves many indexed repositories at once. What stops a SEARCH scoped to repo A from ever returning a symbol belonging to repo B? Trace the READ/search path (not just writes): where the repo scope is applied when querying symbols/vectors/graph. Look at sutra/core/retrieval/pipeline.py, channels, sutra/core/graph/sql_reader.py, vector_store, and the MCP server search entrypoint.' },
  { slug:'channel_fusion_rrf', desc:'Sutra runs several independent retrieval channels (exact-name/moniker, BM25 keyword, vector similarity) for one query and must merge their separate ranked lists into one ordering. Trace how the separate lists are combined (reciprocal rank fusion). Look at sutra/core/retrieval/fusion.py and how pipeline.py calls the channels then fuses.' },
  { slug:'body_hash_skip', desc:'When re-indexing, a symbol whose code body did not change should keep its existing vector rather than being re-embedded. Trace how sutra decides a symbol is unchanged and skips recomputing its embedding. Look at sutra/core/incremental_updater.py (body_hash comparison) and where embedding is skipped.' },
  { slug:'lsp_kickoff', desc:'A call from file A to a function defined in file B needs cross-file resolution. Trace WHEN/HOW sutra starts the language server (LSP) during indexing to resolve such cross-file references, and how the LSP resolver is invoked vs the heuristic resolver. Look at sutra/core/resolver/lsp_resolver.py, sutra/core/resolver/heuristic.py and where the indexer kicks resolution off.' },
  { slug:'unresolved_relationships', desc:'The indexer records that function A calls B, but B may be a library function or defined in a file not indexed. Trace how sutra represents/stores a call (or other relationship) whose target it cannot pin to a known in-repo symbol (unresolved / dangling edge). Look at the extractor relationship emission, the resolver, and how sql_writer stores an unresolved target.' },
]

const results = await pipeline(
  CANDIDATES,
  (c) => agent(scoutPrompt(c), { label:`scout|${c.slug}`, phase:'Scout', schema:GOLD_SCHEMA, effort:'high' })
           .then(g => g ? ({...g, slug:c.slug}) : ({ slug:c.slug, error:true }))
)

return results.filter(Boolean)
