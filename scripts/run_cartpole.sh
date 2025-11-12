#!/usr/bin/env bash
set -e
python -m train.train_cartpole --config configs/cartpole.yaml "$@"
