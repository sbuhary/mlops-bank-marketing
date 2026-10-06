install:
	pip install -r requirements.txt

prepare:
	python -m src.data.load_data

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

pipeline:
	dvc repro

api:
	uvicorn app.main:app --reload

test:
	python -m pytest

docker:
	docker compose up --build
