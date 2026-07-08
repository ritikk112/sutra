SHELL := /bin/bash

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: ui-install ui-build ui-run ui

ui-install:
	cd frontend/web && npm install

ui-build:
	cd frontend/web && npm run build

# Serve the web frontend on :8000 (matches README, frontend/README.md and the
# `sutra ui` command). Keep this port in sync with the docs.
ui-run:
	source .venv/bin/activate && uvicorn frontend.api.main:app --host 127.0.0.1 --port 8000

# Convenience alias mirroring the `sutra ui` CLI command.
ui: ui-run
