.PHONY: install lint test infra-up infra-down

install:
	python -m pip install -e '.[dev]'

lint:
	ruff check .
	ruff format --check .
	mypy src

test:
	pytest

infra-up:
	docker compose up -d

infra-down:
	docker compose down

