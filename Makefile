# Windows: run the retail-rag commands directly (see README).
PY ?= python

.PHONY: setup ingest ask serve eval test lint format up down
setup:   ## install with dev extras
	$(PY) -m pip install --upgrade pip && $(PY) -m pip install -e ".[dev,vertex,openai]"
ingest:  ## build or update the index
	retail-rag ingest
ask:     ## example question
	retail-rag ask "Kiddie pool for the backyard under $$25"
serve:   ## API + UI on http://127.0.0.1:8000
	retail-rag serve
eval:    ## golden-set evaluation with quality gate
	retail-rag eval --min-hit-at-5 0.8 --min-ood-refusal 0.85
test:
	$(PY) -m pytest
lint:
	ruff check src tests && ruff format --check src tests
format:
	ruff check --fix src tests && ruff format src tests
up:      ## Postgres + ingestion + API in Docker
	docker compose up --build
down:
	docker compose down
