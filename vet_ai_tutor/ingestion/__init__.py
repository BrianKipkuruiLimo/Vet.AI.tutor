"""Developer-side ingestion pipeline.

- loaders:    read PDF/PPTX/DOCX/text files into page- or slide-level Documents
- provenance: whitelist and source-catalog validation (fail closed)
- indexer:    chunking, opening/adding to the vector store, atomic index builds
- rebuild:    full rebuild of the index from data/raw_notes/
- staging:    reviewed JSON records from the harvester and document ingestor

The scripts in scripts/ are thin command-line wrappers around these modules.
"""
