#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"

docker run \
--name='tfops_aug' \
--rm \
--tty \
--user "$(id -u):$(id -g)" \
--volume $DIR/../:/src \
-w=/src \
tfops_aug \
python3 -m unittest discover -s tfops_aug/tests/ -p *_test.py