#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"

docker run \
--gpus 0 \
--name='tfops_aug' \
--rm \
--tty \
--user "$(id -u):$(id -g)" \
--volume $DIR/../utils:/src \
tfops_aug \
python3 src/classification_example.py