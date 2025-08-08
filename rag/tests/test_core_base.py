import pytest

from core.base import Document, ProcessingResult, Parser, Embedder, VectorStore, Pipeline, Component


class DummyParser(Parser):
    def parse(self, source: str) -> ProcessingResult:
        doc = Document(content=f"parsed:{source}", id="doc-1", metadata={"k": "v"})
        return ProcessingResult(documents=[doc])


class DummyEmbedder(Embedder):
    def embed(self, texts):
        return [[float(len(t))] for t in texts]


class DummyVectorStore(VectorStore):
    def __init__(self):
        super().__init__(name="DummyVectorStore")
        self.added = []

    def add_documents(self, documents):
        self.added.extend(documents)
        return True

    def search(self, query: str, top_k: int = 10):
        return []

    def delete_collection(self) -> bool:
        return True


class FailingComponent(Component):
    def process(self, documents):
        raise RuntimeError("boom")


def test_document_to_dict_round_trip():
    doc = Document(content="hello", metadata={"a": 1}, id="id-1", source="src")
    d = doc.to_dict()
    assert d["content"] == "hello"
    assert d["metadata"] == {"a": 1}
    assert d["id"] == "id-1"
    assert d["source"] == "src"


def test_pipeline_run_with_source_and_components():
    pipe = Pipeline()
    vs = DummyVectorStore()
    pipe.add_component(DummyParser()).add_component(DummyEmbedder()).add_component(vs)

    result = pipe.run(source="input-data")

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.content == "parsed:input-data"
    assert doc.embeddings == [float(len("parsed:input-data"))]
    # Vector store received the document
    assert vs.added and vs.added[0] is doc
    assert result.errors == []


def test_pipeline_requires_parser_when_source_is_provided():
    pipe = Pipeline()
    pipe.add_component(DummyEmbedder())
    with pytest.raises(ValueError):
        pipe.run(source="x")


def test_pipeline_requires_source_or_documents():
    pipe = Pipeline()
    with pytest.raises(ValueError):
        pipe.run()


def test_pipeline_continues_after_component_error():
    pipe = Pipeline()
    pipe.add_component(FailingComponent(name="WillFail"))
    pipe.add_component(DummyEmbedder())

    docs = [Document(content="abc")]
    result = pipe.run(documents=docs)

    # Error captured and processing continued for subsequent component
    assert any(err.get("component") == "WillFail" for err in result.errors)
    assert result.documents[0].embeddings == [3.0]


def test_vector_store_process_reports_metrics():
    vs = DummyVectorStore()
    docs = [Document(content="a"), Document(content="bb")]
    result = vs.process(docs)
    assert result.metrics.get("stored_count") == 2

