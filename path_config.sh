#!/usr/bin/env bash
# =============================================================================
# Centralized Path Configuration
# =============================================================================
# Change DATA_ROOT here to customize paths for your environment.
# All experiment and plot scripts source this file to get the default data path.
# You can override by setting DATA_ROOT environment variable before running:
#   DATA_ROOT=/your/custom/path ./run_supra.sh

DATA_ROOT="${DATA_ROOT:-/hyperai/input/input0/MAGB_Dataset}"
export DATA_ROOT