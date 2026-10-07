.PHONY: test bench fetch up down paper

test:
	pytest -q

bench:
	python3 -m src.bench.run_all

fetch:
	python3 scripts/fetch_live_data.py

up:
	docker compose up --build

down:
	docker compose down
