devenv:
	pip install -r requirements.txt
format:
	find . -name "*.py" | xargs isort
	black -t py36 .
	mypy --ignore-missing-imports .

debug: format
	python src/train.py --config configs/debug.yaml

tensorboard:
	tensorboard --logdir results
