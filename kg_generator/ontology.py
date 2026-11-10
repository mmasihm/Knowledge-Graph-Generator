"""Creates the ontology once, then persists it so every later batch extends the same schema."""

import json
import logging
from pathlib import Path

from graphrag_sdk import Attribute, AttributeType, Ontology
from graphrag_sdk.models import GenerativeModel
from graphrag_sdk.source import AbstractSource

logger = logging.getLogger(__name__)

# Steers the LLM towards a small, domain-agnostic schema: the corpus mixes history,
# medicine and literature, and generic types let entities from different documents
# (and future batches) connect instead of living in isolated sub-schemas.
ONTOLOGY_BOUNDARIES = """
Build a general-purpose ontology that can describe historical, scientific, cultural
and literary documents. Prefer broad, reusable entity types (for example Person,
Location, Event, Organization, Disease, Artifact, CreativeWork, Concept, Period)
over document-specific ones, and use at most 15 entity types. Every entity must have
a unique, required string attribute called "name". Use UPPER_SNAKE_CASE relation
labels with clear direction (for example PARTICIPATED_IN, LOCATED_IN, CAUSED).
"""


class OntologyStore:
    """Saves the ontology as JSON so it survives restarts and can be extended later."""

    def __init__(self, path: Path) -> None:
        """Remember where the ontology file lives."""
        self._path = path

    def load(self) -> Ontology | None:
        """Return the saved ontology, or None if none exists yet."""
        if not self._path.exists():
            return None
        return Ontology.from_json(self._path.read_text(encoding="utf-8"))

    def save(self, ontology: Ontology) -> None:
        """Write the ontology to disk as pretty-printed JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(ontology.to_json(), indent=2), encoding="utf-8")


def load_or_create_ontology(
    store: OntologyStore, sources: list[AbstractSource], model: GenerativeModel
) -> Ontology:
    """Reuse the saved ontology if there is one; otherwise ask the LLM to derive it from the sources."""
    ontology = store.load()
    if ontology is not None:
        logger.info("Loaded saved ontology (%d entities, %d relations)", len(ontology.entities), len(ontology.relations))
        return ontology

    logger.info("Creating ontology from %d sources (first chunk of each)", len(sources))
    ontology = Ontology.from_sources(sources=sources, model=model, boundaries=ONTOLOGY_BOUNDARIES, hide_progress=True)
    ensure_unique_keys(ontology)
    store.save(ontology)
    logger.info("Saved ontology (%d entities, %d relations)", len(ontology.entities), len(ontology.relations))
    return ontology


def ensure_unique_keys(ontology: Ontology) -> None:
    """Give every entity a unique 'name' attribute if the LLM forgot one.

    The SDK MERGEs nodes on their unique attributes; an entity type without one
    would collapse all of its instances into a single node.
    """
    for entity in ontology.entities:
        if not any(attribute.unique for attribute in entity.attributes):
            logger.warning("Entity %s had no unique attribute; adding 'name'", entity.label)
            entity.attributes.insert(0, Attribute("name", AttributeType.STRING, unique=True, required=True))
