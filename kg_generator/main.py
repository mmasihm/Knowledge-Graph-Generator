"""Entry point: build (or extend) the knowledge graph from the PDFs in DATA_DIR."""

import logging
import sys
from functools import partial
from pathlib import Path

from graphrag_sdk import KnowledgeGraph, KnowledgeGraphModelConfig

from kg_generator.config import ConfigError, Settings, load_settings
from kg_generator.graph_builder import GraphBuilder
from kg_generator.ingestion_tracker import IngestionTracker
from kg_generator.llm import create_model
from kg_generator.ontology import OntologyStore, load_or_create_ontology
from kg_generator.pdf_source import PdfSource
from kg_generator.report import log_graph_report

logger = logging.getLogger("kg_generator")


def configure_logging(level: str) -> None:
    """Log to stdout and silence chatty third-party loggers."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    for noisy in ("pdfminer", "LiteLLM", "httpx", "graphrag_sdk", "unstructured"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def find_pdfs(data_dir: Path) -> list[Path]:
    """Return the PDFs in data_dir in a stable (sorted) order."""
    return sorted(path for path in data_dir.glob("*.pdf") if path.is_file())


def run(settings: Settings) -> int:
    """Run the full pipeline and return a process exit code."""
    pdf_paths = find_pdfs(settings.data_dir)
    if not pdf_paths:
        logger.error("No PDFs found in %s", settings.data_dir)
        return 1
    logger.info("Found %d PDFs in %s", len(pdf_paths), settings.data_dir)

    model = create_model(settings)
    make_source = partial(PdfSource, strategy=settings.pdf_strategy, max_chars=settings.chunk_max_chars)
    ontology = load_or_create_ontology(
        OntologyStore(settings.ontology_path), [make_source(path) for path in pdf_paths], model
    )
    knowledge_graph = KnowledgeGraph(
        name=settings.graph_name,
        model_config=KnowledgeGraphModelConfig.with_model(model),
        ontology=ontology,
        host=settings.falkordb_host,
        port=settings.falkordb_port,
    )

    builder = GraphBuilder(knowledge_graph, IngestionTracker(settings.tracker_path), make_source)
    report = builder.ingest(pdf_paths)
    log_graph_report(knowledge_graph.graph)
    return 0 if not report.failed else 2


def main() -> int:
    """Load settings, configure logging and run the pipeline."""
    try:
        settings = load_settings()
    except ConfigError as error:
        logging.basicConfig()
        logger.error("Configuration error: %s", error)
        return 1
    configure_logging(settings.log_level)
    return run(settings)


if __name__ == "__main__":
    sys.exit(main())
