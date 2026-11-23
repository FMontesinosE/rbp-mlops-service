# Project Rules
- Output code and concise technical notes only. Zero conversational filler or recaps.
- Layout: `src/rbp_mlops/` package, `tests/`, `scripts/`, `docker/`, `.github/workflows/`.
- Python 3.11+, strict type hints (`mypy --strict`), `ruff` formatting, Pydantic v2, Google docstrings on public functions only.
- Never use random/mock tensors in `src/`. Training and tests must parse real 10-column ENCODE narrowPeak BED + FASTA files.
- Ban AI-cliché prose in `README.md` ("delve", "tapestry", "comprehensive", "cutting-edge", "leverage"). Include a Mermaid architecture diagram, `make` commands, and a real benchmark table.
- After generating files, execute `ruff`, `mypy`, and `pytest` via bash and fix every error before stopping.
