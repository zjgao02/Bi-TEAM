#!/bin/bash

INPUT_DATA="${1:-./data/Rezai.csv}"
OUTPUT_NAME="${2:-inference_results.csv}"
MODEL_CHECKPOINT="${3:-/data/home/luruiqiang/guchunbin/Bi-TEAM/checkpoints/best_model_seed_42.pt}"
DEVICE="${4:-cuda:0}"

python ./scripts/inference.py \
  --input_data_path "$INPUT_DATA" \
  --mapping_data_path "./data/ncaa.xlsx" \
  --output_dir './inference_results' \
  --output_name "$OUTPUT_NAME" \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --device "$DEVICE" \
  --batch_size 1 \
  --esm_model "facebook/esm2_t6_8M_UR50D" \
  --chem_model "ibm/materials.selfies-ted" \
  --freeze_esm --freeze_chem 
