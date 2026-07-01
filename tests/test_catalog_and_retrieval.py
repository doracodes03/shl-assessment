import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import Agent
from app.catalog import SHL_PRODUCT_CATALOG_PATH, CatalogItem, load_catalog, SAMPLE_CATALOG_PATH
from app.llm import LLMClient
from app.retrieval import CatalogIndex, SearchFilters
from app.safety import validate_recommendations
from app.schema import Message, Recommendation


def test_load_catalog_filters_packaged_solutions():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    names = [i.name for i in items]
    assert "Entry Level Customer Serv-Retail & Contact Center" in names
    # No bundled "Solution"-named packages should survive the filter.
    assert all("solution" not in n.lower() for n in names)


def test_test_type_codes():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    opq = next(i for i in items if i.entity_id == "720")
    assert opq.test_type_str == "P"
    multi = next(i for i in items if i.entity_id == "3933")  # Customer Service Phone Simulation
    assert set(multi.test_type_codes) == {"B", "S"}


def test_search_returns_relevant_java_items():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    index = CatalogIndex(items)
    results = index.search("Java developer programming knowledge test")
    top_names = [r.item.name for r in results[:3]]
    assert any("Java" in n for n in top_names)


def test_search_duration_filter_excludes_long_items():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    index = CatalogIndex(items)
    results = index.search("coding simulation", filters=SearchFilters(max_duration_minutes=20))
    for r in results:
        assert r.item.duration_minutes is None or r.item.duration_minutes <= 20


def test_search_test_type_filter():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    index = CatalogIndex(items)
    results = index.search("leadership personality", filters=SearchFilters(test_types=["P"]))
    assert results
    for r in results:
        assert "P" in r.item.test_type_codes


def test_find_by_name_loose_match():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    index = CatalogIndex(items)
    matches = index.find_by_name("OPQ32r")
    assert matches and "OPQ32r" in matches[0].name


def test_load_catalog_recovers_from_newlines_inside_strings(tmp_path):
    malformed = (
        '[{"entity_id":"1","name":"Microsoft \n365 (New)",'
        '"link":"https://example.com","description":"desc","job_levels":[],'
        '"languages":[],"duration":"","remote":"yes","adaptive":"no","keys":[]}]'
    )
    path = tmp_path / "catalog.json"
    path.write_text(malformed, encoding="utf-8")

    items = load_catalog(path)

    assert len(items) == 1
    assert items[0].name == "Microsoft 365 (New)"


def test_load_catalog_supports_the_full_shl_catalog_file():
    items = load_catalog(SHL_PRODUCT_CATALOG_PATH)
    assert len(items) > 100


def test_validate_recommendations_accepts_pydantic_http_urls():
    item = CatalogItem(entity_id="1", name="Test", url="https://example.com")
    rec = Recommendation(
        entity_id="1",
        name="Test",
        url="https://example.com/test",
        test_type="K",
        reason="Matches",
        confidence="high",
    )

    assert validate_recommendations([rec], [item]) is True


def test_agent_can_answer_assessment_questions_from_catalog():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    index = CatalogIndex(items)
    agent = Agent(index, llm=LLMClient())

    response = agent.handle([Message(role="user", content="What is OPQ32r?")])

    assert response.recommendations is not None
    assert response.recommendations[0].name == "Occupational Personality Questionnaire OPQ32r"
    assert "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/" in response.reply


def test_agent_explains_when_assessment_is_missing_from_catalog():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    index = CatalogIndex(items)
    agent = Agent(index, llm=LLMClient())

    response = agent.handle([Message(role="user", content="What is a Rust-specific SHL assessment?")])

    assert "could not find" in response.reply.lower()
    assert response.recommendations is None


def test_agent_does_not_ask_redundant_followup_when_role_and_purpose_are_already_present():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    index = CatalogIndex(items)
    agent = Agent(index, llm=LLMClient())

    response = agent.handle([Message(role="user", content="We need a selection battery for senior leadership.")])

    assert "clarify" not in response.reply.lower()
    assert response.recommendations is not None


def test_agent_handles_assessment_followup_questions_without_clarification():
    items = load_catalog(SAMPLE_CATALOG_PATH)
    index = CatalogIndex(items)
    agent = Agent(index, llm=LLMClient())

    response = agent.handle([Message(role="user", content="What is the difference between OPQ and OPQ MQ Sales Report?")])

    assert response.recommendations is None
    assert "could not find" in response.reply.lower()
