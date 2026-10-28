#!/usr/bin/env bash
# Explicit local shared-resource download; run only when this dataset is needed.
set -eu
python "$(dirname "$0")/../../../../../tools/local_resources.py" fetch imagenet-sample
