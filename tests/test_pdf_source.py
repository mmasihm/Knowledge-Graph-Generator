from pathlib import Path

import pytest
from unstructured.documents.elements import NarrativeText, Title

from kg_generator.pdf_source import EmptyPdfError, PdfSource, drop_back_matter

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_sample_pdf_is_extracted_with_unstructured():
    source = PdfSource(SAMPLE_PDF)

    documents = list(source.load())

    assert len(documents) == source.chunk_count == 1
    assert documents[0].id == "sample.pdf#0"
    assert documents[0].content.startswith("Source document: sample")
    assert "Odysseus returns home to Ithaca" in documents[0].content


def test_back_matter_is_dropped():
    elements = [Title("History"), NarrativeText("Facts."), Title("References"), NarrativeText("[1] Cite.")]

    assert [element.text for element in drop_back_matter(elements)] == ["History", "Facts."]


def test_long_text_is_split_into_bounded_chunks():
    paragraphs = [NarrativeText(f"Sentence number {i} about the voyage. " * 10) for i in range(20)]
    source = PdfSource(Path("big.pdf"), max_chars=1000, partition=lambda **_: paragraphs)

    documents = list(source.load())

    assert source.chunk_count == len(documents) > 1
    assert all(len(doc.content) <= 1000 + len("Source document: big\n\n") for doc in documents)


def test_pdf_without_text_raises():
    source = PdfSource(Path("scanned.pdf"), partition=lambda **_: [NarrativeText("   ")])

    with pytest.raises(EmptyPdfError):
        list(source.load())


def test_nothing_is_read_before_load():
    calls = []
    PdfSource(Path("lazy.pdf"), partition=lambda **kwargs: calls.append(kwargs) or [])

    assert calls == []
