#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"

docker run \
--gpus 0 \
--name='auto-augment' \
--rm \
--tty \
--user "$(id -u):$(id -g)" \
--volume $DIR/../:/src \
auto-augment \
python3 src/classification_example.py