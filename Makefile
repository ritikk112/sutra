SHELL := /bin/bash

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: ui-install ui-build ui-run

ui-install:
	cd frontend/web && npm install

ui-build:
	cd frontend/web && npm run build

ui-run:
	source .venv/bin/activate && uvicorn frontend.api.main:app --host 127.0.0.1 --port 8000
