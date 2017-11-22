#!/usr/bin/env bash
export JOB_NAME="facialkeypoint_$(date +%Y%m%d_%H%M%S)"

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir gs://ml-facial-keypoints/$JOB_NAME \
    --package-path ./trainer \
    --module-name trainer.keras-model \
    --region us-central1 \
    -- \
    --train-file gs://ml-facial-keypoints/data/training.pickle
