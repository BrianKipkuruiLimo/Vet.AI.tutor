#!/usr/bin/env python3
"""Web Harvester — Scrape whitelisted URLs and extract structured veterinary data.

Scrapes web pages from approved domains, uses an Ollama LLM to extract
structured information (Taxonomy, Biology, Diseases, etc.), and saves
clean JSON files to a staging area for review before ingestion.

Usage:
    # Scrape a single URL:
    python scripts/harvester.py --url "https://en.wikipedia.org/wiki/Giraffe" --species giraffe

    # Scrape from a URL list file (one URL per line):
    python scripts/harvester.py --url-file urls.txt --species giraffe

    # Scrape with a custom species category:
    python scripts/harvester.py --url "https://example.edu/cattle" --species cattle --category domestic

Prerequisites:
    1. Ollama running:   ollama serve
    2. LLM model:        ollama pull qwen3:1.7b
    3. pip install requests beautifulsoup4
"""

import os
import sys
import json
import hashlib
import re
import argparse
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vet_ai_tutor.config import LLM_MODEL  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
HARVEST_LOG = os.path.join(PROJECT_ROOT, "data", "staging", "harvest_log.json")


# Whitelist checks live in the ingestion package; re-exported for callers and tests.
from vet_ai_tutor.ingestion.provenance import check_domain, load_whitelist  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Web scraping
# ---------------------------------------------------------------------------
def scrape_url(url: str) -> str:
    """Fetch a URL and extract clean text using BeautifulSoup."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("Missing dependencies. Install: pip install requests beautifulsoup4")
        sys.exit(1)

    if urlparse(url).scheme != "https":
        logger.error("Only HTTPS sources are allowed: %s", url)
        return ""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; VetAITutor-Harvester/1.0; "
            "+https://github.com/your-org/vet-ai-tutor)"
        )
    }

    logger.info("Fetching: %s", url)
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return ""

    # Do not follow a redirect to a host outside the approved list.
    try:
        if check_domain(resp.url, load_whitelist()) not in {"tier_1", "tier_2"}:
            logger.error("Blocked redirect to non-whitelisted URL: %s", resp.url)
            return ""
    except RuntimeError as e:
        logger.error("Cannot validate final URL: %s", e)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script, style, nav, footer, header elements
    for tag in soup(["script", "style", "noscript", "template", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    remove_hidden_elements(soup)

    # Try to find the main content area
    main = soup.find("main") or soup.find("article") or soup.find("div", {"id": "content"})
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    # Truncate very long pages to avoid overwhelming the LLM
    max_chars = 12000
    if len(clean_text) > max_chars:
        clean_text = clean_text[:max_chars] + "\n\n[... Content truncated for processing ...]"

    logger.info("Extracted %d characters of text.", len(clean_text))
    return clean_text


# ---------------------------------------------------------------------------
# Prompt-injection defences (OWASP LLM01)
# ---------------------------------------------------------------------------
_HIDDEN_STYLE_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?![.\d])|opacity\s*:\s*0(?![.\d])",
    re.IGNORECASE,
)

# Phrases that address an AI model rather than a veterinary reader.
_INJECTION_RE = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above) (?:instructions|prompts?)"
    r"|disregard (?:all |the )?(?:previous|prior|above)"
    r"|(?:you are|act as) (?:now )?(?:an? )?(?:ai|assistant|language model|chatbot)"
    r"|system prompt|new instructions\s*:|<\s*/?\s*(?:system|context|instructions?)\s*>",
    re.IGNORECASE,
)


def remove_hidden_elements(soup) -> int:
    """Drop elements a human reader cannot see; they are a common injection vector."""
    hidden = [
        tag for tag in soup.find_all(True)
        if tag.has_attr("hidden")
        or str(tag.get("aria-hidden", "")).lower() == "true"
        or _HIDDEN_STYLE_RE.search(str(tag.get("style", "")))
    ]
    for tag in hidden:
        if not tag.decomposed:
            tag.decompose()
    return len(hidden)


def find_injection_phrases(text: str) -> list[str]:
    """Return text snippets that look like instructions aimed at an AI model."""
    warnings = []
    for match in _INJECTION_RE.finditer(text or ""):
        start, end = max(0, match.start() - 60), min(len(text), match.end() + 60)
        warnings.append(text[start:end].replace("\n", " "))
    return warnings


# ---------------------------------------------------------------------------
# LLM-based structured extraction
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are a veterinary data extraction assistant. Read the following raw text
scraped from a web page about a specific animal species. Extract the information
into the following structured JSON format.

RULES:
- Only extract information that is explicitly stated in the text.
- If a section has no relevant information, set its value to null.
- Keep each section concise (max 300 words per section).
- Do NOT invent or hallucinate information not present in the source text.
- Return ONLY valid JSON, no markdown code fences, no extra text.

OUTPUT FORMAT:
{{
    "common_name": "string",
    "scientific_name": "string or null",
    "taxonomy": "string or null — classification details: kingdom, phylum, class, order, family, genus, species",
    "biology": "string or null — anatomy, physiology, reproduction, lifespan, habitat",
    "development": "string or null — growth stages, maturation, life cycle",
    "diseases": "string or null — common diseases, symptoms, diagnosis, treatment",
    "nutrition": "string or null — diet, feeding, nutritional requirements",
    "behavior": "string or null — social behavior, communication, migration",
    "conservation": "string or null — conservation status, threats, population trends",
    "veterinary_notes": "string or null — clinical notes, handling, anesthesia, common procedures",
    "summary": "A brief 2-3 sentence summary of the animal based on the text"
}}

RAW TEXT:
{raw_text}

JSON OUTPUT:"""


def extract_structured_data(raw_text: str, species: str, llm_model: str = LLM_MODEL) -> dict | None:
    """Use Ollama LLM to extract structured veterinary data from raw text."""
    try:
        from vet_ai_tutor.llm_adapter import build_llm
    except ImportError:
        logger.error("langchain-ollama not installed. Install: pip install langchain-ollama")
        return None

    prompt = EXTRACTION_PROMPT.format(raw_text=raw_text)

    logger.info("Sending text to LLM (%s) for structured extraction...", llm_model)
    try:
        llm = build_llm(llm_model, num_predict=2048, num_ctx=8192, temperature=0.1)
        response = llm.invoke(prompt)
    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return None

    # Parse the JSON response
    try:
        # Try to find JSON in the response (handle markdown fences)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
        data = json.loads(text)
        logger.info("Successfully extracted structured data.")
        return data
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s", e)
        logger.debug("Raw LLM response:\n%s", response[:500])
        return None


# ---------------------------------------------------------------------------
# Staging area management
# ---------------------------------------------------------------------------
def save_to_staging(
    structured_data: dict,
    species: str,
    category: str,
    source_url: str,
    source_tier: str,
    source_title: str = "",
    author: str = "",
    journal: str = "",
    publication_year: str = "",
    raw_source_text: str = "",
) -> str:
    """Save extracted data to the staging area as a JSON file for human review."""
    os.makedirs(STAGING_DIR, exist_ok=True)

    # Create a unique filename based on species + URL hash
    url_hash = hashlib.md5(source_url.encode()).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{species}_{url_hash}_{timestamp}.json"
    filepath = os.path.join(STAGING_DIR, filename)

    injection_warnings = find_injection_phrases(raw_source_text) + find_injection_phrases(
        json.dumps(structured_data, ensure_ascii=False)
    )
    if injection_warnings:
        logger.warning("Possible prompt-injection text in %s — review before approval.", source_url)

    # Build the staging document
    staging_doc = {
        "status": "pending_review",
        "injection_warnings": injection_warnings,
        "species": species.lower(),
        "category": category.lower(),
        "source_url": source_url,
        "source_tier": source_tier,
        "source_domain": urlparse(source_url).hostname,
        "source_title": source_title or source_url,
        "author": author,
        "journal": journal,
        "publication_year": str(publication_year),
        "raw_source_text": raw_source_text,
        "date_harvested": datetime.now(timezone.utc).isoformat(),
        "extracted_data": structured_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(staging_doc, f, indent=2, ensure_ascii=False)

    logger.info("Saved to staging: %s", filepath)
    return filepath


def update_harvest_log(url: str, species: str, status: str, filepath: str = ""):
    """Append an entry to the harvest log for tracking."""
    os.makedirs(os.path.dirname(HARVEST_LOG), exist_ok=True)

    entry = {
        "url": url,
        "species": species,
        "status": status,
        "file": os.path.basename(filepath) if filepath else "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log = []
    if os.path.exists(HARVEST_LOG):
        with open(HARVEST_LOG, encoding="utf-8") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []

    log.append(entry)

    with open(HARVEST_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main harvesting pipeline
# ---------------------------------------------------------------------------
def harvest_url(
    url: str,
    species: str,
    category: str = "unknown",
    llm_model: str = LLM_MODEL,
    source_title: str = "",
    author: str = "",
    journal: str = "",
    publication_year: str = "",
) -> bool:
    """Full pipeline: validate domain → scrape → extract → stage."""

    # 1. Whitelist check
    whitelist = load_whitelist()
    tier = check_domain(url, whitelist)
    if tier not in {"tier_1", "tier_2"}:
        logger.warning(
            "BLOCKED: %s is not in the approved whitelist. "
            "Add its HTTPS domain to data/whitelist.json after review.",
            urlparse(url).hostname,
        )
        update_harvest_log(url, species, "blocked_whitelist")
        return False

    tier = tier or "unverified"
    logger.info("Domain tier: %s", tier)

    # 2. Scrape
    raw_text = scrape_url(url)
    if not raw_text:
        update_harvest_log(url, species, "scrape_failed")
        return False

    # 3. LLM extraction
    structured = extract_structured_data(raw_text, species, llm_model)
    if structured is None:
        # Save raw text as fallback
        structured = {
            "common_name": species,
            "raw_text": raw_text[:5000],
            "summary": f"Raw unstructured text about {species} (LLM extraction failed).",
            "_extraction_failed": True,
        }
        logger.warning("LLM extraction failed — saving raw text as fallback.")

    # 4. Save to staging
    filepath = save_to_staging(
        structured, species, category, url, tier, source_title, author,
        journal, publication_year, raw_text,
    )
    update_harvest_log(url, species, "staged", filepath)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Harvest veterinary data from approved web sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", type=str, help="Single URL to scrape.")
    parser.add_argument("--url-file", type=str, help="Path to a file with one URL per line.")
    parser.add_argument("--species", type=str, required=True, help="Animal species name (e.g., giraffe).")
    parser.add_argument(
        "--category", type=str, default="unknown",
        choices=["wild", "domestic", "exotic", "aquatic", "avian", "unknown"],
        help="Animal category.",
    )
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL, help="Ollama model for extraction.")
    parser.add_argument("--source-title", default="", help="Title of the cited source.")
    parser.add_argument("--author", default="", help="Source author(s); verify before approval.")
    parser.add_argument("--journal", default="", help="Journal or publisher.")
    parser.add_argument("--year", dest="publication_year", default="", help="Publication year.")
    args = parser.parse_args()

    if not args.url and not args.url_file:
        parser.error("Provide --url or --url-file.")

    urls = []
    if args.url:
        urls.append(args.url)
    if args.url_file:
        with open(args.url_file) as f:
            urls.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))

    success = 0
    failed = 0
    for url in urls:
        ok = harvest_url(url, args.species, args.category, args.llm_model,
                         args.source_title, args.author, args.journal, args.publication_year)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Harvesting complete: {success} succeeded, {failed} failed")
    print(f"Staged files are in: {os.path.abspath(STAGING_DIR)}")
    print("Review them, then run: python scripts/ingest_staged.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
