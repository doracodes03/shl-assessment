# Assessment knowledge and conversation capability plan

## Goal
Make the SHL agent behave more like the reference conversations by combining:
- grounded recommendations from the catalog,
- accurate assessment-specific explanations,
- explicit handling of missing catalog items,
- correct SHL product links,
- no hallucinated facts.

## Sources of truth
The agent must only answer using:
1. The loaded catalog JSON records.
2. The assessment product URLs from the catalog.
3. The conversation history and current user request.
4. Optional LLM summarization only for phrasing, never for inventing facts.

## Required capabilities

### 1. Assessment knowledge lookup
The agent should be able to answer questions such as:
- What is OPQ32r?
- What is the difference between OPQ and OPQ MQ Sales Report?
- Is Graduate Scenarios suitable for graduates?
- What does the Contact Center Call Simulation measure?

It should do this by matching the user’s question to catalog items by name or semantic similarity and returning only facts that are supported by the catalog metadata.

### 2. Recommendation reasoning
The agent should reason about why a recommendation fits a role, including:
- role level,
- purpose (selection vs development),
- skill or industry cues,
- language constraints,
- duration constraints,
- assessment type needs (cognitive, personality, simulation, knowledge, etc.).

### 3. Correct links
Every recommendation and assessment knowledge response should include the correct SHL product URL when the assessment exists in the catalog.

### 4. Missing-item handling
If an assessment is not present in the catalog, the agent should clearly explain that it cannot find a matching catalog item and explicitly say what is missing.

### 5. No hallucination
The agent must not invent:
- assessment purpose,
- duration,
- languages,
- test type,
- or differences between assessments.

If the evidence is missing, it should say so plainly.

## Conversation patterns to support
The implementation should cover the patterns seen in the 10 sample conversations:
- Clarifying role, level, language, and purpose.
- Recommending a battery for a role or hiring need.
- Comparing two assessments.
- Explaining why one assessment is better than another.
- Handling “missing” assessments such as Rust-specific tests.
- Supporting follow-up questions about a specific assessment.
- Respecting when the user asks for legal/compliance guidance and redirecting to the right source.

## Implementation approach
- Extend the catalog item model with a richer text representation for factual retrieval.
- Add a lightweight assessment knowledge retrieval layer that looks up catalog items by name and by semantic similarity.
- Add a response mode for assessment Q&A.
- Route follow-up questions to that mode when they are about a specific assessment.
- Keep all factual content grounded in catalog metadata and links.
