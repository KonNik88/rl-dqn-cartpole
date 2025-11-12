#!/usr/bin/env bash
set -e
python -m train.train_pong --config configs/pong.yaml "$@"
