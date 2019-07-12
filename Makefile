setup:
	pip install -r requirements.txt

format:
	find . -name "*.py" | xargs isort
	black -t py36 .
	mypy --ignore-missing-imports .

debug: format
	python src/train.py --config configs/debug.yaml

fetch:
	./fetch.sh labo

start-fetch:
	watch -n 1 make fetch

deploy:
	./deploy.sh labo

start-deploy:
	watch -n 1 make deploy

smi:
	nvidia-smi -l 3

tb:
	tensorboard --logdir results
