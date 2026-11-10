from kg_generator.ingestion_tracker import IngestionTracker, file_sha256


def test_hash_depends_only_on_content(tmp_path):
    first, second, other = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")
    other.write_bytes(b"different")

    assert file_sha256(first) == file_sha256(second)
    assert file_sha256(first) != file_sha256(other)


def test_unknown_hash_is_not_ingested(tmp_path):
    assert not IngestionTracker(tmp_path / "ledger.json").is_ingested("abc")


def test_marked_file_is_remembered_across_instances(tmp_path):
    ledger = tmp_path / "state" / "ledger.json"
    IngestionTracker(ledger).mark_ingested("abc", "Titanic.pdf")

    reloaded = IngestionTracker(ledger)

    assert reloaded.is_ingested("abc")
    assert reloaded.entries()["abc"]["file_name"] == "Titanic.pdf"


def test_write_leaves_no_temporary_file(tmp_path):
    ledger = tmp_path / "ledger.json"
    IngestionTracker(ledger).mark_ingested("abc", "a.pdf")

    assert [path.name for path in tmp_path.iterdir()] == ["ledger.json"]
