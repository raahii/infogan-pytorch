#!/bin/bash
host=${1:-labo}

rsync -auvz \
      --delete \
      --exclude='.DS_Store' \
      --exclude='.git*' \
      --exclude='.python-version' \
      --exclude='__pycache__' \
      --exclude='data' \
      --exclude='result' \
      --exclude='deploy.sh' \
      --exclude='.mypy_cache' \
      ~/study/infogan/ $host:~/study/infogan/
