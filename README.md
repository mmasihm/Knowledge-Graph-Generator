# Knowledge Graph Generator

Turns a folder of PDFs into a knowledge graph in FalkorDB. Text is extracted with
**Unstructured**, an LLM (Gemini via **LiteLLM**) derives an ontology and then extracts
entities and relations through **GraphRAG-SDK**, and the result is queryable with Cypher.
The pipeline is incremental: a later batch of PDFs extends the same graph and ontology.

## Architecture

```
kg_generator/
  config.py             Settings from env / .env (key never logged)
  pdf_source.py         PdfSource(AbstractSource): Unstructured partition -> drop back matter -> chunk by title
  ontology.py           Create the ontology once with the LLM, persist it as JSON, reuse it afterwards
  llm.py                LiteLLM model whose chat sessions retry transient errors with backoff
  ingestion_tracker.py  SHA-256 ledger of ingested files (atomic JSON writes)
  graph_builder.py      Ingest PDFs one at a time; skip known files; isolate failures
  report.py             Graph summary + example Cypher queries
  main.py               Wiring (composition root)
```

Flow: `main` lists the PDFs → loads the saved ontology or builds one from the first
chunk of each PDF → `GraphBuilder` hashes each file, skips it if the ledger already has
the hash, otherwise hands a `PdfSource` to `KnowledgeGraph.process_sources`, which
MERGEs entities and relations into FalkorDB → the file is recorded in the ledger → a
summary is logged.

Design notes:
- **Extending the SDK, not forking it.** `PdfSource` subclasses the SDK's `AbstractSource`
  and yields `Document`s, so the SDK's ontology and extraction steps use it unchanged.
- **Low memory.** One PDF is processed at a time (the SDK materialises all chunks of
  the sources it is given), files are hashed in 1 MB blocks, and chunks are generated lazily.
- **Idempotent.** Files are identified by content hash, not name; the ledger is written
  after each file, so an interrupted run resumes where it stopped. Entities are MERGEd
  on their unique `name`, so re-processing a file does not duplicate nodes.
- **Ready for more batches.** The ontology is saved to `state/ontology.json` (and to the
  graph's `_schema` graph by the SDK) and is reused, so a new batch is extracted into the
  same schema. Pointing `DATA_DIR` at a new folder is all that is needed to add documents.
- **Signal over noise.** "See also / References / Bibliography / External links" sections
  are dropped before chunking: in these Wikipedia-style PDFs they are 40–70 % of the
  text but contain no new facts. This cuts LLM cost and spurious entities.
- **Failure handling.** A corrupt or text-less PDF is logged and skipped, not recorded, and
  retried on the next run. Rate limits, timeouts and 5xx errors are retried with
  exponential backoff; auth errors fail fast. If some chunks of a file fail, the file is
  recorded with the failed chunk ids; if all fail, it is not recorded.
- **SOLID, kept practical.** One responsibility per module; `GraphBuilder` receives its
  graph, tracker and source factory (dependency injection), and depends on a small
  `SourceProcessor` protocol rather than the concrete SDK class, which keeps tests offline.

## Why these tools

| Tool | Why |
|---|---|
| **GraphRAG-SDK 0.8.2** | Required framework. Its ontology → extraction → FalkorDB pipeline is exactly the job. Pinned to 0.8.2 because it has the `AbstractSource` / `Ontology` / `KnowledgeGraph` / `KnowledgeGraphModelConfig` API this project extends; 1.x is an async rewrite that drops those classes and pulls in GLiNER and PyTorch. |
| **Unstructured** (`partition_pdf`, `chunk_by_title`) | Returns typed elements (Title, NarrativeText, ListItem…) instead of a raw text dump. That lets us cut back matter by heading and chunk along section boundaries, so each LLM call sees one coherent topic. The `fast` strategy (pdfminer) suits these digital PDFs; `hi_res`/OCR can be enabled via `PDF_STRATEGY` for scans. |
| **FalkorDB** | Required graph store; native backend of GraphRAG-SDK. In-memory with low query latency, speaks Cypher, ships a browser UI on port 3000. |
| **Gemini 3.5 Flash via LiteLLM** | Strong at structured JSON extraction, long context, low cost per token, fast (2.5 Flash is no longer offered to new API users). LiteLLM makes the provider a config value: set `LLM_MODEL=openai/gpt-4.1` (or any LiteLLM id) without code changes. |
| **Docker / Compose** | One command starts FalkorDB (with a health check) and the app. CPU-only PyTorch keeps the image several GB smaller than the default CUDA build. |
| **pytest** | Offline unit tests with fakes for the LLM and FalkorDB. |

## Run it

Requires Docker with Compose v2.

```bash
cp .env.example .env        # then set LLM_API_KEY (a Gemini key: https://aistudio.google.com/apikey)
```

```bash
docker compose up --build
```

The app ingests `data/batch_1/*.pdf`, logs a graph summary with example queries, and
exits; FalkorDB keeps running (Ctrl+C to stop it). Run `docker compose up` again: the
saved ontology is reused and every file is skipped. PDFs must be readable by the
container's non-root user; an unreadable file is reported as "No text extracted".

Explore the graph at http://localhost:3000 (graph `knowledge_graph`) or with Cypher:

```bash
docker compose exec falkordb redis-cli GRAPH.QUERY knowledge_graph "MATCH (n) RETURN labels(n)[0], count(n)"
```

See the generated ontology and the ingestion ledger:

```bash
docker compose run --rm app cat state/ontology.json state/ingested_files.json
```

Start from scratch (removes the graph, ontology and ledger together):

```bash
docker compose down -v
```

### Configuration

All settings are environment variables (see `.env.example`): `LLM_API_KEY` (required),
`LLM_MODEL` (default `gemini/gemini-3.5-flash`), `GRAPH_NAME`, `FALKORDB_HOST`,
`FALKORDB_PORT`, `DATA_DIR` (default `data/batch_1`), `STATE_DIR`, `PDF_STRATEGY`
(`fast`), `CHUNK_MAX_CHARS` (`8000`), `LOG_LEVEL`.

## Tests

```bash
docker compose run --rm --no-deps app pytest -q
```

Or locally (Python 3.12): `pip install -r requirements.txt -r requirements-dev.txt`, then
`pytest -q`. The PDF test needs NLTK data (`punkt_tab`, `averaged_perceptron_tagger_eng`),
which the Docker image already contains. The LLM and FalkorDB are faked, so no key,
database or network is needed (the suite passes with `--network none`).

## Known limitations

- **Ontology from a sample.** It is derived from the first chunk of each PDF (an SDK
  behaviour), steered towards general types. Types seen only later in a document may be
  missed; extraction then ignores entities that fit no type.
- **Entity resolution is name-based.** Nodes merge on exact `name`, so "Titanic" and "RMS
  Titanic" can become two nodes. A normalisation or embedding-based dedup pass would fix it.
- **No provenance on nodes.** GraphRAG-SDK 0.8.2 does not record which chunk a fact came
  from, so facts cannot be traced back to a page.
- **LLM non-determinism.** Re-running on an empty graph yields a
  slightly different ontology and graph.
- **Throughput.** Batch 1 is 113 chunks after trimming. The SDK rate-limits itself to 15
  LLM calls/minute per process, so a first run takes roughly 8–10 minutes; extraction with
  Unstructured is ~50 s for all five PDFs. Large corpora would need a queue and parallel workers.
- **Ontology pass re-reads PDFs.** The SDK only uses the first chunk of each source for the
  ontology, but Unstructured must still partition the whole PDF to produce it (~1 minute
  for batch 1). This happens once; afterwards the saved ontology is reused.
- **Ledger and graph are separate stores.** They share a Compose volume lifecycle, but
  wiping only FalkorDB leaves the ledger claiming files are ingested
  (`docker compose down -v` resets both).
- **SDK side effects.** The SDK writes per-chunk debug logs to `logs/` and uses `print`
  for unknown entity labels; both are inside the library.
- **Image size.** `unstructured[pdf]` depends on PyTorch even though the `fast` strategy
  never uses it; the image is about 2 GB even with CPU-only wheels.
