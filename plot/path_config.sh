#!/usr/bin/env bash
# =============================================================================
# Centralized Path Configuration for Plot Scripts
# =============================================================================
# Change DATA_ROOT here to customize paths for your environment.
# All plot scripts source this file to get the default data path.
# You can override by setting DATA_ROOT environment variable before running:
#   DATA_ROOT=/your/custom/path ./plot_gnn.sh

DATA_ROOT="${DATA_ROOT:-/hyperai/input/input0/MAGB_Dataset}"
export DATA_ROOT