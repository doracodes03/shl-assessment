# Approach Document — SHL Assessment Recommendation Agent

## Architecture

A stateless FastAPI service (`GET /health`, `POST /chat`) wraps a single
`Agent` class that re-derives all conversation state from the full
`messages` history on every call (no DB, no session store — matches the
spec's stateless contract). The agent is a tool-use loop against Claude
(Anthropic Messages API): the model never sees or invents catalog data
directly — it calls `search_catalog` / `compare_assessments` to look
things up, and `propose_shortlist` to commit to specific catalog
`entity_id`s. The server resolves those ids to `{name, url, test_type}`
itself before they reach the response, so **the model can never fabricate
a URL or recommend something outside the catalog** — that guarantee lives
in code, not in prompting, which is what the "items from catalog only"
hard-eval actually requires.

Catalog ingestion is a separate, offline step (`scripts/fetch_catalog.py`)
that writes a normalized JSON cache to disk. The running service only ever
reads that cache — it never makes a network call mid-conversation. This
keeps every `/chat` call well under the 30s timeout regardless of upstream
availability, and means a catalog refresh (re-run the script, redeploy) is
decoupled from request latency.

## Catalog ingestion & scope filtering

Each catalog row carries `keys` (categories), which I map to the single-
letter `test_type` codes used in the response schema: K=Knowledge & Skills,
P=Personality & Behavior, A=Ability & Aptitude, S=Simulations,
B=Biodata & Situational Judgment, C=Competencies, D=Development & 360. This
mapping was reverse-engineered from the labeled traces (C1–C3) and cross-
checked against the catalog data directly.

**Individual Test Solutions vs. Pre-packaged Job Solutions:** the public
JSON dump doesn't carry an explicit tab/category field for this split. I
approximate it with a conservative name heuristic (excludes bundled
products whose name contains the standalone word "Solution(s)", e.g.
"Entry Level Cashier Solution", "Customer Service Phone Solution") plus a
hand-maintainable `data/catalog_overrides.json` allow/deny list, so
misclassifications can be fixed without a code change. This is the single
biggest known gap — see "What didn't work" below.

## Retrieval

BM25 (rank-bm25) over `name + description + job_levels + keys`, combined
with **structured filters** (job level, max duration, language, test
type) that the model extracts from conversation context and passes as
tool arguments. I chose lexical search over embeddings deliberately:

- SHL product names/descriptions are short and jargon-dense ("OPQ32r",
  "SVAR", "Verify G+"); query terms tend to literally match catalog
  vocabulary, so BM25 recall is already strong.
- The traces show the *hard* part isn't semantic similarity — it's
  disambiguating via hard constraints (English-US vs. UK accent variant,
  25-minute cap, "selection" vs. "development" report format). BM25 can't
  do that alone, hence the filter layer doing the real precision work.
- No embedding API / vector store dependency, so it runs on any free tier
  without extra infra (per the assignment's resource list).

## Prompt design

One system prompt encodes all four required behaviors (clarify / recommend
/ refine / compare) plus scope enforcement (refuse general hiring/legal
advice, refuse and ignore prompt-injection attempts embedded in user
text) and an explicit instruction to ground every claim in tool output,
never prior knowledge. Four tools: `search_catalog`, `compare_assessments`,
`propose_shortlist`, `mark_complete`. `end_of_conversation` is only ever
true when the model explicitly calls `mark_complete` — never inferred from
text — so it can't drift out of sync with what's actually in
`recommendations`.

Refinement (C1, "Selection — comparing candidates against a leadership
benchmark") and comparison (asking the difference between two simulation
products, as in C3 turn 4) are handled by the same loop: the model is
instructed to re-search and re-propose rather than restart, and to answer
comparisons only from `compare_assessments` output.

## Evaluation approach

1. **Schema/hard-eval checks** are structural, not LLM-dependent: Pydantic
   validates `recommendations` is `None` or 1–10 items; every
   `Recommendation` is built server-side from a real catalog item, so
   non-catalog URLs are categorically impossible.
2. **Retrieval unit tests** (`tests/test_catalog_and_retrieval.py`, no
   network/LLM needed) check that the packaged-solution filter holds, the
   `test_type` code mapping matches the traces, and that job-level /
   duration / test-type filters actually constrain results — these run in
   CI on every change.
3. **Trace replay**: I treat C1–C3 as integration fixtures — replaying
   each trace's user turns through `/chat` and checking the final
   shortlist's `entity_id`s against the labeled expected set gives a
   directly comparable Recall@10 signal before submitting.
4. **Behavior probes** (turn-1 vagueness, off-topic refusal, refinement
   honored) are exercised by short scripted conversations in the same
   replay harness style the grader uses, so failures surface locally
   first.

## What didn't work / open issues

- Treating "Solution" name-matching as the IT-vs-Job-Solutions filter is a
  heuristic, not ground truth — it will misclassify any individual test
  whose name happens to contain "solution(s)" verbatim, and won't catch
  packaged bundles that don't use that word. The override file is the
  mitigation, not a fix; a future pass should scrape the catalog website's
  two tabs directly to get this split authoritatively.
- I initially considered embeddings for retrieval but dropped them after
  noticing BM25 alone matched the traces' expected items for the
  technical/knowledge-test queries; the gain wasn't worth the added infra
  and cost for queries that are mostly literal keyword matches.
- AI tools used: this codebase (FastAPI service, retrieval, prompt/tool
  design, tests) was built with Claude via agentic coding assistance for
  scaffolding and iteration speed; all design decisions and the retrieval
  approach above were reviewed and adjusted by hand against the traces.
