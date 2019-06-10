devenv:
	pip install -r requirements.txt

fmt:
	find . -name "*.py" | xargs isort
	black -t py36 .
	mypy --ignore-missing-imports .

debug: fmt
	python src/train.py --config configs/default.yaml
