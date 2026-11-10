"""Remembers which PDFs (by content hash) are already in the graph."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HASH_BLOCK_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    """Hash a file in fixed-size blocks so large PDFs never load fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


class IngestionTracker:
    """JSON ledger of ingested files keyed by SHA-256, so re-runs skip unchanged PDFs."""

    def __init__(self, ledger_path: Path) -> None:
        """Load the ledger from disk, or start empty if it does not exist yet."""
        self._path = ledger_path
        self._entries: dict[str, dict] = self._read()

    def is_ingested(self, file_hash: str) -> bool:
        """Return True if a file with this hash was already ingested."""
        return file_hash in self._entries

    def mark_ingested(self, file_hash: str, file_name: str) -> None:
        """Record a fully ingested file and persist the ledger immediately."""
        self._entries[file_hash] = {
            "file_name": file_name,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._write()

    def entries(self) -> dict[str, dict]:
        """Return a copy of all ledger entries."""
        return dict(self._entries)

    def _read(self) -> dict[str, dict]:
        """Read the ledger file if present."""
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self) -> None:
        """Write atomically (temp file + rename) so a crash cannot corrupt the ledger."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")
        os.replace(temp_path, self._path)
