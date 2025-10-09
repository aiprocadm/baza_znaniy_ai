SHELL := /bin/bash
PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip
APP_MODULE := app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
IMAGE ?= kb-ai:local

.PHONY: install lint format test run worker build clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	if [ -f requirements-dev.txt ]; then $(PIP) install -r requirements-dev.txt; fi

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
