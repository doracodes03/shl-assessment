# SHL Assessment Recommendation Agent

A conversational agent that turns a vague hiring need into a grounded
shortlist of SHL Individual Test Solutions, exposed as a stateless
FastAPI service (`GET /health`, `POST /chat`).

## Project layout

```
app/
  schema.py     Pydantic request/response models (matches the API spec)
  catalog.py    Loads + normalizes + filters the catalog JSON
  retrieval.py  BM25 + structured-filter search over the catalog
  llm.py        Thin wrapper around the Anthropic Messages API (tool use)
  agent.py      System prompt, tool loop, shortlist/refusal/compare logic
  main.py       FastAPI app (/health, /chat)
scripts/
  fetch_catalog.py   Pulls the live catalog JSON -> data/catalog.json
data/
  catalog_sample.json     Small bundled fixture (used if catalog.json absent)
  catalog_overrides.json  Manual allow/deny list for the IT-Solutions filter
tests/
  test_catalog_and_retrieval.py   No network/LLM required
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python scripts/fetch_catalog.py        # refresh data/catalog.json from the live catalog
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Without running `fetch_catalog.py`, the service falls back to the bundled
`data/catalog_sample.json` (20 hand-picked items) so it's runnable out of
the box for development.

## Run tests

```bash
pytest tests/ -q
```

The included tests cover catalog filtering and retrieval only (no network
or LLM credentials required). Agent/LLM behavior is best validated against
the provided conversation traces by replaying them through `/chat`.

## Try it

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders"}]}'
```

## Deploying

Any ASGI-friendly free tier works (Render, Fly, Railway, Modal, HF Spaces).
Set `ANTHROPIC_API_KEY` (or swap `app/llm.py` for a different free-tier
provider) and run `scripts/fetch_catalog.py` as a build step / nightly cron
so `/chat` never makes a network call mid-conversation.

See `APPROACH.md` for design rationale, evaluation approach, and known
limitations.
