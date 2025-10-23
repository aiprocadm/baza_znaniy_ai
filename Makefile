SHELL := /bin/bash
PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip
APP_MODULE := app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
IMAGE ?= kb-ai:local
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

.PHONY: venv install dev lint format test run worker up down migrate seed build clean \
        web-install web-lint web-format web-test web-build web-run

venv:
	$(PYTHON) -m venv .venv

install:
	$(PIP) install --upgrade pip
	$(PIP) install --index-url $(TORCH_INDEX) torch==2.4.1
	$(PIP) install -r requirements-runtime.txt -r requirements-llm.txt

dev: install
	$(PIP) install -r requirements-dev.txt

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check . --fix

test:
	pytest -q

run:
        uvicorn $(APP_MODULE) --factory --host $(HOST) --port $(PORT)

worker:
        python -m app.worker.main

up:
        docker compose up -d --build

down:
        docker compose down --remove-orphans

migrate:
        alembic upgrade head

seed:
        python -m backend.app.db.seed

build:
        docker build -t $(IMAGE) .

clean:
        rm -rf __pycache__ */__pycache__ .pytest_cache cov_html .coverage coverage.xml

web-install:
        cd frontend && npm install

web-lint:
        cd frontend && npm run lint

web-format:
        cd frontend && npm run format

web-test:
        cd frontend && npm run test

web-build:
        cd frontend && npm run build

web-run:
        cd frontend && npm run dev -- --host 0.0.0.0
