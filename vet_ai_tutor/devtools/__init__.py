"""Developer-dashboard logic (no Streamlit imports except in ui.py).

- paths:   every file location the dashboard reads or writes
- jobs:    long tasks (rebuild, OCR, evaluation, ...) as detached processes
- catalog: metadata review for data/raw_notes (draft -> approved catalog)
- status:  read-only summaries: index, evaluation history, release state
- ui:      shared Streamlit widgets (job panel)
"""
