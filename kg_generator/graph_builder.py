"""Feeds PDFs into the knowledge graph one file at a time, skipping files already ingested."""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kg_generator.ingestion_tracker import IngestionTracker, file_sha256
from kg_generator.pdf_source import PdfSource

logger = logging.getLogger(__name__)


class SourceProcessor(Protocol):
    """The part of graphrag_sdk.KnowledgeGraph the builder needs (eases testing and swapping)."""

    failed_documents: list[str]

    def process_sources(self, sources: list, instructions: str | None = None, hide_progress: bool = False) -> None:
        """Extract entities and relations from the sources into the graph."""


@dataclass
class IngestionReport:
    """Outcome of one ingestion run, by file name."""

    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


class GraphBuilder:
    """Ingests PDFs sequentially so only one document is ever held in memory."""

    def __init__(
        self,
        knowledge_graph: SourceProcessor,
        tracker: IngestionTracker,
        make_source: Callable[[Path], PdfSource],
    ) -> None:
        """Inject the graph, the ledger of ingested files and the source factory."""
        self._graph = knowledge_graph
        self._tracker = tracker
        self._make_source = make_source

    def ingest(self, pdf_paths: Iterable[Path]) -> IngestionReport:
        """Ingest every new PDF; already-ingested files are skipped, broken ones are logged and skipped."""
        report = IngestionReport()
        for path in pdf_paths:
            file_hash = file_sha256(path)
            if self._tracker.is_ingested(file_hash):
                logger.info("Skipping %s (already ingested)", path.name)
                report.skipped.append(path.name)
            elif self._ingest_file(path, file_hash):
                report.ingested.append(path.name)
            else:
                report.failed.append(path.name)
        logger.info("Ingestion done: %d ingested, %d skipped, %d failed",
                    len(report.ingested), len(report.skipped), len(report.failed))
        return report

    def _ingest_file(self, path: Path, file_hash: str) -> bool:
        """Extract one PDF into the graph and record it; return False if it must be retried later."""
        logger.info("Ingesting %s", path.name)
        source = self._make_source(path)
        try:
            # One source per call: the SDK materialises all chunks of the sources it is given.
            self._graph.process_sources([source], hide_progress=True)
        except Exception as error:  # a single bad PDF must never abort the whole run
            logger.error("Skipping %s: %s: %s", path.name, type(error).__name__, error)
            logger.debug("Traceback for %s", path.name, exc_info=True)
            return False

        failed_chunks = sorted(self._graph.failed_documents or [])
        if source.chunk_count == 0:
            logger.error("Skipping %s: it produced no text chunks", path.name)
            return False
        if failed_chunks:
            # Not recorded, so the next run retries the whole file; MERGE keeps that free of duplicates.
            logger.error("Not recording %s: %d of %d chunks failed (will retry next run): %s",
                         path.name, len(failed_chunks), source.chunk_count, failed_chunks)
            return False

        self._tracker.mark_ingested(file_hash, path.name)
        logger.info("Ingested %s (%d chunks)", path.name, source.chunk_count)
        return True
