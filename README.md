# Vet AI Tutor

Vet AI Tutor is an offline veterinary learning assistant. It uses retrieval-augmented generation (RAG): a local knowledge index supplies reviewed source passages to a local Ollama model, which answers only from those passages and cites them as `[S1]`, `[S2]`. The model is not fine-tuned on the ingested data, and nothing needs the internet at runtime.

## Hardware and model choice

The defaults are tuned for a modest laptop CPU with no GPU. Reference machine:

| Component | Spec |
|---|---|
| CPU | Intel Core i7-7600U (2 cores / 4 threads, 2.8 GHz) |
| RAM | 16 GB (about 7 GB free while running) |
| GPU | Intel HD 620 integrated, not used by Ollama |

On that machine, with the default **`qwen3:1.7b`** model (1.4 GB):

| Measurement | Result |
|---|---|
| Model load | about 3 s |
| First words of an answer | about 9 s |
| Full answer | 15–40 s, depending on how much context is retrieved |
| Repeated question | instant (served from the cache) |

The app turns off Qwen3's "thinking" mode, which would otherwise add hundreds of hidden tokens to every answer and make CPU replies several times slower.

| Model | Size | When to use it |
|---|---|---|
| `qwen3:1.7b` (default) | 1.4 GB | Best balance on a 2-core CPU; follows citation rules well |
| `qwen3:4b` | 2.5 GB | Better answers at about half the speed; needs about 4 GB of free RAM |
| `qwen3:0.6b` | 0.5 GB | Very old or low-RAM machines; weaker at sticking to the sources |

Switch models with `VET_LLM_MODEL`, for example `VET_LLM_MODEL=qwen3:4b python -m vet_ai_tutor.api`. If answers are cut short, raise `VET_LLM_NUM_PREDICT` (default 256 tokens). If retrieved passages don't fit in the context window, raise `VET_LLM_NUM_CTX` (default 2048). Larger values for either use more RAM and make answers slower.

## Quick start

```bash
bash scripts/setup_env.sh        # uses uv if installed, otherwise pip
source .venv/bin/activate
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
python -m vet_ai_tutor.api
```

Open `http://localhost:8000`. The server listens on localhost only. To serve other devices on a classroom LAN, set `VET_API_HOST=0.0.0.0` and add their address to `VET_CORS_ORIGINS`.

Or run `bash scripts/start.sh`, which also checks that Ollama and the index are ready and opens the chat page when the server is up. Each browser tab keeps its own conversation.

Streamlit is used only for the developer dashboard (see below); it is a dev dependency and is not shipped to students.

## How a question is answered

1. **Short follow-ups are expanded.** A question like "how is it treated?" has the previous question added in front of it before searching.
2. **Two searches run.** Vector search (`nomic-embed-text`) finds passages with similar meaning. BM25 keyword search finds exact terms such as drug or pathogen names.
3. **The results are fused** with Reciprocal Rank Fusion, so a strong keyword match is not crowded out by vector results.
4. **The top `k` passages (default 5) go to the model**, marked as reference text rather than instructions. The model must cite them, or reply "Not found in my knowledge base."
5. **The answer streams to the browser** as Server-Sent Events. The sources are sent in the same response, so the LLM runs once per question.

## Environments

Knowledge collection, scraping, review, and index creation are developer tasks. Offline users receive a built application, the Ollama runtime and models, and a vetted knowledge index. They ask questions and may install a **signed** knowledge patch. They never ingest source documents.

## Developer dashboard

Most developer work is done in the dashboard. It runs locally and is not part of the student app:

```bash
uv run streamlit run dev_dashboard.py      # http://localhost:8502
```

`.streamlit/config.toml` binds it to `127.0.0.1`. Keep it that way: the dashboard edits the source catalog, rebuilds the index and signs releases.

| Page | What you do there |
|---|---|
| **Sources** | Review the drafted metadata for every lecture file: filter by course code (parsed from file names, for example `VMED 344`), edit cells, set a field for a whole course at once, preview a file's extracted text, and approve every ready draft in the view with one click. Approval runs the rebuild's own checks; the catalog is backed up before each change. Uploading documents here saves them to `data/raw_notes/uploaded/` and drafts their metadata. |
| **Index** | Dry run (about a minute, no embedding), full rebuild, OCR of scanned PDFs, install hints for Tesseract and LibreOffice, and the last rebuild report (unreadable, scanned, unapproved files). |
| **Web staging** | Harvest a whitelisted page, check and correct each staged record's provenance, see injection warnings, approve, and ingest approved records. |
| **Evaluation** | Run the retrieval evaluation and the answer benchmark (optionally with RAGAS), and see recall, MRR, grounding and speed over time. |
| **Quiz** | Generate draft questions, review each one next to its source passage (approve, reject, or choose the correct option), and publish to the bank. |
| **Release** | Signing-key status and fingerprint, key generation, patch creation, created patches, and recent activity. |

Long tasks (rebuild, OCR, evaluation, quiz generation, patch creation) run as **background jobs**. They keep running if you close the browser tab, the page shows live progress and the full output, and they can be cancelled. Jobs that write the index (rebuild, ingest, create patch) can't run at the same time. Job records are kept in `data/jobs/`.

Every action is also available as a command, as described below.

## Developer ingestion workflow

### Local documents (lecture notes, PDFs, slides)

1. Put the files in `data/raw_notes/`.
2. Draft metadata entries automatically:

   ```bash
   python scripts/draft_source_metadata.py --course "BVM" --default-author "Your institution"
   ```

   This reads titles, authors and years from each file's properties and its filename, and writes them to `data/source_metadata.draft.json`. Running it again only adds new files; existing drafts and your edits are kept (`--fresh` starts over). Every draft is marked `"_needs_review": true` and has an empty `source_tier`, so it can't be indexed by accident.
3. Review each draft:
   - check `source_title`, `author`, `journal` (the course or publisher) and `publication_year`;
   - set `source_tier` to `tier_1` or `tier_2`;
   - delete `_needs_review` and `_detected`;
   - move the entry into `data/source_metadata.json`.

   You can approve files in batches. Files without a reviewed entry are skipped and listed in the rebuild output.
4. Check what will be indexed, then rebuild:

   ```bash
   python scripts/rebuild_vectorstore.py --dry-run   # fast: no embedding
   python scripts/rebuild_vectorstore.py
   ```

   The dry run lists files that are approved, not yet approved, unreadable, scanned (need OCR) or contain no text. The rebuild builds the new index beside the current one, keeps a backup, and swaps it in. It also writes `index_info.json`, which records the embedding model and the prefixes the index was built with. The full report is saved to `archive/rebuild_summary.json`.

   Embedding is the slow step: about 2–3 chunks per second on the reference laptop, so the full set of notes (about 30,000 chunks) takes several hours the first time. Close other heavy programs such as browsers while it runs. Vectors are cached in `data/cache/embeddings/`, so later rebuilds only embed new or changed chunks, and an interrupted rebuild resumes where it stopped.

#### How documents are read

| Format | Reader | What is extracted |
|---|---|---|
| PDF | pypdfium2 | Text per page, keeping bullet line breaks; about 15× faster than pypdf on these notes |
| PPTX | python-pptx | Text boxes, tables (`cell \| cell`), grouped shapes and speaker notes, per slide |
| DOCX | python-docx | Paragraphs and tables in document order |
| `.doc` / `.ppt` | LibreOffice, then the above | Converted once into `data/cache/converted/`; the original file and its catalog entry are unchanged |
| Damaged PPTX/DOCX | Direct XML read | Text is recovered even when embedded images are corrupt or the archive is truncated |
| CSV, TXT, MD, JSON | Built in | Rows as `column: value`; text as is |

Converting legacy Office files needs LibreOffice's document components (the plain `libreoffice-core` package can't open documents):

```bash
sudo apt install libreoffice-writer-nogui libreoffice-impress-nogui
```

**Scanned PDFs.** Pages with no text layer are listed under "Needs OCR". OCR them once on the developer machine (requires `sudo apt install tesseract-ocr`):

```bash
python scripts/ocr_scanned_pdfs.py               # PDFs that are mostly scanned
python scripts/ocr_scanned_pdfs.py --all-empty   # also image-only pages inside text PDFs
```

OCR text is cached in `data/cache/ocr/`, keyed by file content, so the original PDFs are never modified, and the next rebuild picks it up. Only pages where Tesseract's mean word confidence is at least 70% (`VET_OCR_MIN_CONFIDENCE`) are indexed. Handwritten pages typically score much lower and would only add gibberish, so they are reported but skipped.

> **Rebuild required after this upgrade.** `nomic-embed-text` expects the prefixes `search_query:` / `search_document:`, which the app now applies. An index built before this change has no `index_info.json`; the app still works with it, but shows a warning and retrieves less accurately until you rebuild.

A full rebuild replaces the whole index, including records added through staged ingestion. Re-run `ingest_staged.py` for those records after a rebuild.

### Web sources

`data/whitelist.json` fails closed: only HTTPS URLs on listed tier 1 or tier 2 domains are allowed, including after redirects. The harvester removes hidden HTML (a common way to plant instructions for an AI) and flags any text that looks like a command to a model:

```bash
python scripts/harvester.py --url https://approved.example/vet-topic \
  --species cattle --category domestic \
  --source-title "Source title" --author "Verified author" \
  --journal "Journal or publisher" --year 2025
python scripts/ingest_staged.py --review
python scripts/ingest_staged.py --show staged_record.json
python scripts/ingest_staged.py --approve staged_record.json
python scripts/ingest_staged.py
```

Only approved records with complete provenance and an approved tier are ingested. Records with `injection_warnings` are refused until you remove the flagged text, or confirm it is harmless by setting `"injection_warnings_reviewed": true`. Exact duplicate content is rejected.

The **Web staging** page of the developer dashboard does the same steps with forms.

## Knowledge patches (signed)

Patches are signed with Ed25519, so an offline machine only accepts updates made by the holder of your private key. Checksums alone would not be enough: anyone who can build a ZIP can also compute matching checksums.

Do this once, on the developer machine:

```bash
python scripts/generate_signing_key.py
```

- The private key is written to `~/.config/vet-ai-tutor/patch_signing_key.pem`. Keep it secret and back it up; if it is lost, new patches can't be signed for existing installs.
- The public key is written to `data/patch_signing_public.pem`. Commit it; it ships with the app.

For each release, stop the app and run:

```bash
python scripts/create_data_patch.py --version v2.1
```

Transfer the ZIP from `data/patches/` by USB. On the offline machine, choose **Update Knowledge Base** in the web UI and enter the developer token, or run:

```bash
python scripts/apply_data_patch.py --patch data/patches/vet_ai_data_patch_v2.1.zip
```

The apply step checks, in order:
- the signature;
- that the patch format, vector backend, embedding model and prefixes match this install;
- that every archive path is safe;
- the checksums;
- that every vector carries provenance.

It then stages the new index beside the active one, keeps a backup, and swaps them atomically. Roll back with `python scripts/apply_data_patch.py --rollback`.

If no public key is installed, patches are refused. For local development only, `VET_ALLOW_UNSIGNED_PATCHES=1` together with `create_data_patch.py --unsigned` skips signing. The API update route also requires `VET_ADMIN_TOKEN`.

## Offline deployment bundle (Linux)

Build on an internet-connected machine with the same Linux architecture and Python version as the target. Install the approved Ollama models, generate the signing key, and download Ollama's offline `.deb` installer first:

```bash
python scripts/build_offline_bundle.py \
  --model-store "$HOME/.ollama/models" \
  --ollama-installer ./ollama-linux-amd64.deb \
  --output ./dist/vet-ai-tutor-offline.zip
```

On the offline machine, extract the archive, run `./install_offline.sh`, and start the app with `./run_offline.sh`. The bundle includes the patch public key and creates a private developer token in `.admin_token`; keep that file private. `VET_OFFLINE_ONLY=1` (the default) stops the embedding fallback from reaching the network.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, model, chunk count, cache stats, active sessions, index warnings |
| POST | `/ask` | `{question, k?, remember?, session_id?}` → answer, sources, citations |
| POST | `/ask_stream` | Same input; Server-Sent Events `sources`, `token`…, `done` (or `error`) |
| GET | `/history?session_id=` | One session's conversation |
| POST | `/reset` | `{session_id}`: clear one session's conversation |
| POST | `/followups` | Suggested follow-up questions |
| GET | `/quiz?n=5&topic=` | Random reviewed quiz questions (without answers) |
| POST | `/quiz/check` | `{id, choice}` → correct option, explanation, source |
| GET | `/quiz/topics` | Documents with quiz questions, with counts |
| GET | `/dev/staging`, `/dev/health` | Developer status (`X-Admin-Token` required) |
| GET | `/dev/metrics` | Per-stage timing statistics (`X-Admin-Token` required) |
| POST | `/dev/cache/clear` | Clear cached answers (`X-Admin-Token` required) |
| POST | `/knowledge/update` | Apply a signed patch (`X-Admin-Token` required) |

Every error response has the same shape: `{"error", "detail", "request_id"}`. Question endpoints are rate-limited per client (`VET_RATE_LIMIT_PER_MINUTE`, default 20) to protect a CPU-bound host. Interactive docs are at `/docs`.

## Dependencies

Dependencies are declared in `pyproject.toml` and pinned to exact versions in `uv.lock`, so every developer machine, CI run and offline bundle gets the same packages. `requirements.txt` is generated from the lock, with hashes, for pip installs and the offline bundle builder. Don't edit it by hand:

```bash
uv add "some-package>=1.0"          # or: uv add --group dev some-tool
uv export --format requirements-txt --no-dev --frozen --output-file requirements.txt
```

CI fails if `requirements.txt` and `uv.lock` disagree. Groups:
- the default **dev** group (installed by `uv sync`): pytest, ruff, mypy, pre-commit, and Pillow for OCR;
- the optional **eval** group (`uv sync --group eval`): RAGAS.

Neither group ships to students.

`uv sync` removes packages that aren't in the lock. If you installed extras by hand (for example `langchain-huggingface` for the optional embedding fallback), use `uv sync --inexact`.

## Tests, linting and evaluation

```bash
uv run ruff check .
uv run mypy
uv run pytest -q
```

The tests use fakes and generate their own sample documents, so they need neither Ollama nor an index. CI (`.github/workflows/ci.yml` at the repository root) runs them on Python 3.10, 3.12 and 3.13. To lint on every commit, run `pre-commit install` from the repository root.

### Retrieval quality

Measure after every change to chunking, the embedding model or retrieval settings:

```bash
python scripts/evaluate_retrieval.py                  # hybrid recall@k and MRR
python scripts/evaluate_retrieval.py --mode vector    # compare with vector-only
python scripts/evaluate_retrieval.py --limit 20 --min-recall 0.5   # fails below 50%
```

A QA item in `data/eval/qa_pairs.json` counts as found when a retrieved chunk comes from its `source` file (and `page` or `slide`, if given). Items without `source` fall back to matching words from the answer, which is much less reliable.

To attach sources, someone who knows the material runs:

```bash
python scripts/review_qa_sources.py          # pick the passage that answers each question
python scripts/review_qa_sources.py --add    # write new questions first
```

For each question the tool shows the reference answer and the top retrieved passages; the reviewer presses 1–9 to choose one, `n` if none is right, `m` to type a source, or `s` to skip. Progress is saved after every answer. Aim for 150–200 questions spread across courses.

### Answer quality

```bash
python scripts/benchmark.py --limit 10      # quick check
python scripts/benchmark.py                 # all questions
python scripts/benchmark.py --ragas         # adds RAGAS scores (slow; see below)
```

Every run reports cheap, deterministic checks alongside timing:
- **Declined:** how often the tutor answered "not found".
- **Cited a source:** share of answers that cite at least one `[S#]`.
- **Invalid citations:** answers citing a passage that wasn't retrieved.
- **Key-term recall:** share of the reference answer's terms that appear in the answer.
- **Grounding:** share of the answer's terms that appear in the retrieved passages. Low grounding suggests the model added things its sources don't say; the benchmark prints the unsupported terms.

`--ragas` adds two LLM-judged RAGAS metrics:
- **faithfulness:** are the answer's claims supported by the passages?
- **context recall:** do the passages cover the reference answer?

They use a local judge model (`--judge-model`, default `qwen3:4b`; run `ollama pull qwen3:4b` first) and the eval group (`uv sync --group eval`). With a 1.7B judge it takes about 2.5 minutes per question on the reference laptop, so run it overnight before releases. A small judge is noisy, so compare runs with each other rather than reading scores as absolute. In a sanity check, a correct answer scored faithfulness 1.0 and a fabricated one 0.0.

## Quiz mode

Students press 🧠 in the chat to answer five multiple-choice questions. If they type a topic first, the quiz is limited to that topic. Each answer shows the explanation and the source document and page. Questions come from a **reviewed bank**: generating one takes 20–40 s on a laptop CPU, which is too slow to do live, and generated questions need checking.

```bash
python scripts/generate_quiz.py --limit 20                       # drafts -> data/quiz/pending.json
python scripts/generate_quiz.py --model qwen3:4b --sources mastitis --limit 10
python scripts/review_quiz.py                                    # approve, reject, or fix the answer
python scripts/review_quiz.py --publish                          # -> <index>/quiz_bank.json
```

Generated drafts are rejected automatically when:
- they don't have exactly 4 distinct options;
- an option is "all/none of the above";
- the correct option isn't supported by the source passage;
- a "wrong" option is also supported by the passage, meaning there are two correct answers.

Small models often produce that last problem, so review is still essential; `qwen3:4b` writes noticeably better questions than the default model.

The published bank is stored inside the index directory. It ships with signed knowledge patches and offline bundles, and survives rebuilds. The API never sends answers to the browser; `/quiz/check` grades each choice on the server.

## Configuration

Every setting is a typed field in `vet_ai_tutor/config.py`. Set it with a `VET_`-prefixed environment variable, or in a `.env` file in the project root; environment variables win. To start a `.env` file, copy the documented template, which lists all settings at their defaults:

```bash
cp .env.example .env                    # then uncomment what you change
python -m vet_ai_tutor.config           # show every setting and where its value came from
```

An invalid value stops startup with a message that names the variable, for example `VET_RETRIEVER_K: Input should be less than or equal to 50`. A misspelled name is reported instead of being silently ignored: `Unknown setting VET_LLM_MODLE is ignored — did you mean VET_LLM_MODEL?`. `.env` is gitignored because it may hold `VET_ADMIN_TOKEN`; regenerate the template with `python -m vet_ai_tutor.config --env-example > .env.example` after adding a setting.

The settings you're most likely to change:

| Variable | Default | Meaning |
|---|---|---|
| `VET_LLM_MODEL` | `qwen3:1.7b` | Ollama model for answers |
| `VET_LLM_NUM_CTX` / `VET_LLM_NUM_PREDICT` | `2048` / `256` | Context window and answer length (tokens) |
| `VET_LLM_REASONING` | `0` | Qwen3 thinking mode; very slow on CPU |
| `VET_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model; changing it requires a rebuild |
| `VET_RETRIEVER_K` / `VET_RETRIEVAL_CANDIDATES` | `5` / `20` | Passages sent to the model / candidates per search |
| `VET_CHUNK_SIZE` / `VET_CHUNK_OVERLAP` | `600` / `100` | Characters per chunk; changing them requires a rebuild |
| `VET_API_HOST` / `VET_API_PORT` | `127.0.0.1` / `8000` | Server bind address |
| `VET_ADMIN_TOKEN` | unset | Enables the developer routes |
| `VET_USE_RERANKER` | `0` | Cross-encoder re-ranking (needs `sentence-transformers` and a cached model) |
| `VET_OCR_MIN_CONFIDENCE` | `70` | Minimum Tesseract confidence for an OCR'd page to be indexed |
| `VET_INGEST_CACHE_DIR` | `data/cache` | OCR text, converted legacy files and the embedding cache |
| `VET_LOG_FORMAT` / `VET_LOG_LEVEL` | `text` / `INFO` | `json` writes one JSON object per log line |

The vector backend is Chroma; FAISS remains available as a legacy fallback.

## Performance and logs

Every answered question writes one summary line showing where the time went:

```
question done in 43.2s | retrieval 0.43s (vector 0.43, bm25 0.00, fusion 0.00, rerank 0.00)
  | first token 37.4s (prompt processing 37.0s) | generation 42.8s, 8.1 tok/s
```

On the reference laptop almost all of the wait is **prompt processing**: the model reading the retrieved passages before it writes. Retrieval takes under half a second. To make answers faster, send less context, for example `VET_RETRIEVER_K=3` or smaller chunks, and confirm with the retrieval evaluation that quality holds.

Every log line carries the request ID (also returned in the `X-Request-ID` header), so one question can be followed through the log. With `VET_LOG_FORMAT=json`, lines can be filtered with `jq`, for example `jq 'select(.event == "question") | .stages'`. `GET /dev/metrics` (admin token) returns the median, 95th percentile and maximum of each stage over the last 500 questions, plus the cache hit rate; `/health` includes the medians.
