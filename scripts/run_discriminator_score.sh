#!/bin/bash
# This script scores pianoroll data using a trained discriminator.
# It first looks for the configuration and model parameter files in the 
# experiment directory and then performs scoring with the trained model
# on the specified GPU.
# Usage: run_discriminator_score.sh [EXP_DIR] [INPUT_NPY] [GPU_NUM] [AGG_METHOD]

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

if [ ! -f "$1/config.yaml" ]; then
  echo "Configuration file not found" && exit 1
fi
if [ ! -f "$1/params.yaml" ]; then
  echo "Model parameter file not found" && exit 1
fi
if [ ! -d "$1/model" ]; then
  echo "Model checkpoint directory not found" && exit 1
fi
if [ -z "$2" ]; then
  echo "Input .npy file not specified" && echo "Usage: $0 [EXP_DIR] [INPUT_NPY] [GPU_NUM] [AGG_METHOD]" && exit 1
fi
if [ ! -f "$2" ]; then
  echo "Input file not found at $2" && exit 1
fi

# Set GPU device
if [ -z "$3" ]; then
  gpu="0"
else
  gpu="$3"
fi

# Set aggregation method (default: mean)
if [ -z "$4" ]; then
  agg_method="mean"
else
  agg_method="$4"
fi

# Create results directory
out_dir="$1/results/discriminator_score"
mkdir -p "$out_dir"

# Run the scoring script
python "$DIR/../src/discriminator_score.py" \
  --checkpoint_dir "$1/model" \
  --result_dir "$out_dir" \
  --params "$1/params.yaml" \
  --config "$1/config.yaml" \
  --input_npy "$2" \
  --output_raw "$out_dir/raw_logits.npy" \
  --output_prob "$out_dir/probs.npy" \
  --output_aggregated "$out_dir/aggregated_probs.npy" \
  --output_aggregated_raw "$out_dir/aggregated_raw.npy" \
  --aggregation_method "$agg_method" \
  --gpu "$gpu" 