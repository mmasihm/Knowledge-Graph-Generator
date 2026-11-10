"""GraphRAG-SDK source that reads PDFs with Unstructured."""

import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from graphrag_sdk.document import Document
from graphrag_sdk.source import AbstractSource
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Title
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

# Sections after which an article holds only citations and links: no new facts,
# but many tokens. Everything from the first such heading onward is dropped.
BACK_MATTER_TITLES = frozenset(
    {"see also", "references", "bibliography", "external links", "further reading"}
)

Partitioner = Callable[..., list[Element]]


class EmptyPdfError(ValueError):
    """Raised when a PDF yields no extractable text."""


class PdfSource(AbstractSource):
    """Extracts a PDF with Unstructured and yields one Document per section-aware chunk."""

    def __init__(
        self,
        path: Path,
        strategy: str = "fast",
        max_chars: int = 8000,
        partition: Partitioner = partition_pdf,
    ) -> None:
        """Store the PDF path and extraction options; nothing is read until load()."""
        super().__init__(str(path))
        self.path = Path(path)
        self.strategy = strategy
        self.max_chars = max_chars
        self._partition = partition
        self.chunk_count = 0  # chunks yielded by the most recent load()

    def load(self) -> Iterator[Document]:
        """Yield the PDF body as text chunks, each tagged with its source file."""
        elements = drop_back_matter(self._extract_elements())
        chunks = chunk_by_title(
            elements,
            max_characters=self.max_chars,
            new_after_n_chars=int(self.max_chars * 0.75),
            combine_text_under_n_chars=self.max_chars // 4,
        )
        self.chunk_count = 0
        for chunk in chunks:
            text = chunk.text.strip()
            if text:
                # The file name gives the LLM context to disambiguate entities.
                yield Document(f"Source document: {self.path.stem}\n\n{text}", f"{self.path.name}#{self.chunk_count}")
                self.chunk_count += 1

    def _extract_elements(self) -> list[Element]:
        """Partition the PDF into typed elements, failing loudly if it has no text."""
        elements = self._partition(filename=str(self.path), strategy=self.strategy, languages=["eng"])
        elements = [element for element in elements if element.text and element.text.strip()]
        if not elements:
            raise EmptyPdfError(f"No text extracted from {self.path.name} (unreadable file, scanned PDF or missing NLTK data?)")
        logger.debug("Extracted %d elements from %s", len(elements), self.path.name)
        return elements


def drop_back_matter(elements: Sequence[Element]) -> list[Element]:
    """Return the elements before the first back-matter heading (e.g. 'References')."""
    for index, element in enumerate(elements):
        if isinstance(element, Title) and element.text.strip().lower() in BACK_MATTER_TITLES:
            return list(elements[:index])
    return list(elements)
