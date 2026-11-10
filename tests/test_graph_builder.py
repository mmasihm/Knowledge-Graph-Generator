from pathlib import Path

from unstructured.documents.elements import NarrativeText

from kg_generator.graph_builder import GraphBuilder
from kg_generator.ingestion_tracker import IngestionTracker
from kg_generator.pdf_source import PdfSource


class FakeKnowledgeGraph:
    """Stands in for graphrag_sdk.KnowledgeGraph: no FalkorDB, no LLM."""

    def __init__(self, fail_chunks: bool = False):
        self.fail_chunks = fail_chunks
        self.failed_documents: list[str] = []
        self.processed: list[str] = []

    def process_sources(self, sources, instructions=None, hide_progress=False):
        documents = [document for source in sources for document in source.load()]
        self.processed.extend(document.id for document in documents)
        self.failed_documents = [document.id for document in documents] if self.fail_chunks else []


def fake_partition(**kwargs):
    name = Path(kwargs["filename"]).name
    if name.startswith("corrupt"):
        raise ValueError("broken xref table")
    return [NarrativeText(f"Text of {name}.")]


def make_pdfs(tmp_path, *names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(name.encode())  # distinct content -> distinct hash
        paths.append(path)
    return paths


def build(tmp_path, graph):
    tracker = IngestionTracker(tmp_path / "ledger.json")
    return GraphBuilder(graph, tracker, lambda path: PdfSource(path, partition=fake_partition)), tracker


def test_second_run_skips_already_ingested_files(tmp_path):
    pdfs = make_pdfs(tmp_path, "a.pdf", "b.pdf")
    graph = FakeKnowledgeGraph()
    builder, _ = build(tmp_path, graph)

    first = builder.ingest(pdfs)
    second = builder.ingest(pdfs)

    assert first.ingested == ["a.pdf", "b.pdf"]
    assert second.skipped == ["a.pdf", "b.pdf"] and second.ingested == []
    assert graph.processed == ["a.pdf#0", "b.pdf#0"]


def test_corrupt_pdf_is_skipped_without_stopping_the_run(tmp_path):
    pdfs = make_pdfs(tmp_path, "corrupt.pdf", "good.pdf")
    builder, tracker = build(tmp_path, FakeKnowledgeGraph())

    report = builder.ingest(pdfs)

    assert report.failed == ["corrupt.pdf"]
    assert report.ingested == ["good.pdf"]
    assert [entry["file_name"] for entry in tracker.entries().values()] == ["good.pdf"]


def test_file_is_not_marked_when_every_chunk_failed(tmp_path):
    pdfs = make_pdfs(tmp_path, "a.pdf")
    builder, tracker = build(tmp_path, FakeKnowledgeGraph(fail_chunks=True))

    report = builder.ingest(pdfs)

    assert report.failed == ["a.pdf"]
    assert tracker.entries() == {}
