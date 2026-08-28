.PHONY: install lint test serve evaluate infra-up infra-down

install:
	python -m pip install -e '.[dev]'

lint:
	ruff check .
	ruff format --check .
	mypy src

test:
	pytest

serve:
	uvicorn evidencebench.api:app --reload --port 8080

evaluate:
	evidencebench evaluate --dataset datasets/evaluation.example.jsonl --output artifacts/evaluation.json

infra-up:
	docker compose up -d

infra-down:
	docker compose down
