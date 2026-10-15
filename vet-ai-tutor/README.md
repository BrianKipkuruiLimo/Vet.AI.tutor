# Vet AI Tutor

Vet AI Tutor is a local veterinary learning assistant. It uses retrieval-augmented generation (RAG): a local vector database supplies source passages to a local Ollama model. The model is not fine-tuned on the ingested data.

## Environments

Knowledge collection, scraping, review, and index creation are developer tasks. Offline users receive a built application, Ollama runtime and models, and a vetted knowledge index. They ask questions and may install a signed-off knowledge patch; they do not ingest source documents.

## Developer ingestion workflow

1. Add local PDFs, Word documents, PowerPoint files, text, or curated JSON to `data/raw_notes/`, or harvest an approved HTTPS URL.
2. For local documents, create a reviewed metadata entry in `data/source_metadata.json`. Copy `data/source_metadata.example.json` as a template. Each record needs `source_title`, `author`, `journal` (journal, publisher, institution, or course), `publication_year`, `source_url`, and a reviewed `source_tier` (`tier_1` or `tier_2`). Keys may be relative paths under `data/raw_notes/` or unique filenames.
3. Build the index with `python scripts/rebuild_vectorstore.py`. It checks provenance and the whitelist before it replaces the current index. It stops if any loaded source is uncatalogued or unapproved.

For web sources, `data/whitelist.json` is fail-closed: only HTTPS URLs on listed tier 1 or tier 2 domains are allowed. Scraping stages a JSON record and keeps the fetched text for review:

```bash
python scripts/harvester.py --url https://approved.example/vet-topic \
  --species cattle --category domestic \
  --source-title "Source title" --author "Verified author" \
  --journal "Journal or publisher" --year 2025
python scripts/ingest_staged.py --review
python scripts/ingest_staged.py --show staged_record.json
```

Developers can also use the local Streamlit dashboard for uploads, scraping, per-record review, ingestion, and patch creation:

```bash
streamlit run dev_dashboard.py --server.address 127.0.0.1 --server.port 8502
```

Keep the developer dashboard bound to localhost; it is an ingestion tool, not part of the end-user offline app.

After checking the source and extracted claims, fill any missing or corrected provenance in the JSON, then approve and ingest that individual record:

```bash
python scripts/ingest_staged.py --approve staged_record.json
python scripts/ingest_staged.py
```

Only approved records with complete provenance and an approved tier are eligible. Exact duplicate staged content is rejected. The index stores source title, author, journal/publisher, year, URL, tier, species/topic, and page or slide details where available. Keep the original source documents and reviewed staging records as the audit trail.

The existing bulk rebuild remains a full rebuild; it now requires the reviewed metadata catalog and validates sources before deleting/replacing the current index. Incremental staged ingestion adds reviewed JSON topics to the current index.

## Build a data patch

Stop the application before packaging its index. Patch creation checks that the active index has required provenance, writes a versioned ZIP, and includes SHA-256 checksums for all database files:

```bash
python scripts/create_data_patch.py --version v2.1
```

Transfer the ZIP to the offline machine by USB. From the app, select **Update Knowledge Base** and provide the developer token, or use the command line:

```bash
python scripts/apply_data_patch.py --patch data/patches/vet_ai_data_patch_v2.1.zip
```

The API update route requires `VET_ADMIN_TOKEN`. Applying a patch validates its format, matching vector backend and embedding model, safe archive paths, required index files, and checksums. It stages the replacement beside the active index, keeps a backup, then swaps directories atomically. Roll back with `python scripts/apply_data_patch.py --rollback`. Restart the app after a command-line update.

## Build an offline deployment bundle (Linux)

Build on an internet-connected staging machine for the same Linux architecture and Python version as the offline target. Install the approved Ollama model(s) and obtain Ollama's offline `.deb` installer first. The bundle contains the app, index, Ollama model store, installer, and Python wheelhouse; no downloads are needed on the target:

```bash
python scripts/build_offline_bundle.py \
  --model-store "$HOME/.ollama/models" \
  --ollama-installer ./ollama-linux-amd64.deb \
  --output ./dist/vet-ai-tutor-offline.zip
```

On the offline machine, extract the archive, run `./install_offline.sh`, install Ollama if prompted, and start the app with `./run_offline.sh`. The bundle creates a private developer token in `.admin_token`; keep that file private. The web UI uses system fonts and the app defaults to `VET_OFFLINE_ONLY=1`, which prevents an embedding fallback from reaching the network.

## Local development

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Configure a developer update token before launching the API:

```bash
export VET_ADMIN_TOKEN="choose-a-long-random-secret"
python -m vet_ai_tutor.api
```

Open `http://localhost:8000`. The Streamlit app is also available with `streamlit run vet_ai_tutor_app.py`. Runtime inference and retrieval are local. Web harvesting is a developer-only task and requires a network connection.

Streamlit telemetry is disabled in `.streamlit/config.toml` for offline use.

## Tests and evaluation

```bash
pytest tests/ -v
python scripts/benchmark.py
python scripts/evaluate_retrieval.py
```

Configuration is in `vet_ai_tutor/config.py`; environment variables use the `VET_` prefix. By default the vector backend is Chroma, the embedding model is `nomic-embed-text`, and the LLM is `llama3.2:3b`.
