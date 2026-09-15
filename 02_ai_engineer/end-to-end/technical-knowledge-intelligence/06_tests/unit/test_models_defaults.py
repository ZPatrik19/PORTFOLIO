from tkip.models import AnswerDiagram, AskRequest, Chunk, KnowledgeAnswer


def test_mutable_defaults_are_not_shared_between_model_instances() -> None:
    first = KnowledgeAnswer(answer="a", confidence=1.0)
    second = KnowledgeAnswer(answer="b", confidence=1.0)
    first.used_tools.append("search_library")

    assert second.used_tools == []


def test_request_document_filters_are_not_shared() -> None:
    first = AskRequest(question="What is RAG?")
    second = AskRequest(question="What is BM25?")
    first.document_ids.append("doc-1")

    assert second.document_ids == []


def test_diagram_lists_are_independent() -> None:
    first = AnswerDiagram(title="one")
    second = AnswerDiagram(title="two")
    first.nodes.append({"id": "a", "label": "A"})

    assert second.nodes == []
