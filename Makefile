.PHONY: install data train api dashboard test docker-up clean

install:
	pip install -r requirements.txt

data:
	python src/data/generate_data.py --n-samples 50000
	python src/features/build_features.py

train:
	python src/models/train.py --experiment-name credit_risk_v1

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

mlflow:
	mlflow ui --port 5000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -x -k "not slow"

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

drift-check:
	python src/monitoring/drift_detector.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov coverage.xml

all: install data train test
	@echo "✅ Full pipeline complete!"
