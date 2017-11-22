#!/usr/bin/env bash

gcloud ml-engine local train \
    --package-path ./trainer \
    --module-name trainer.keras-model \
    -- \
    --train-file /Users/joca/Code/ml/data/training.pickle \
    --job-dir ./tmp/ml
