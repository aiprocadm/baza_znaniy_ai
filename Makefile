SHELL := /bin/bash
PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip
APP_MODULE := app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
IMAGE ?= kb-ai:local
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

.PHONY: venv install dev lint format test run worker build clean

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

build:
	docker build -t $(IMAGE) .

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache cov_html .coverage coverage.xml
