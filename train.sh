#!/bin/bash

# Script to run peptide classification model training
# Default arguments
export PYTHONPATH="$PYTHONPATH:$(pwd)"
SEED=42
DEVICE="cuda:0"
BATCH_SIZE=128
EPOCHS=100
LOG_WANDB=True

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --seed)
            SEED="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --log_wandb)
            LOG_WANDB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="python ./scripts/train.py --seed $SEED --device $DEVICE --batch_size $BATCH_SIZE --epochs $EPOCHS --freeze_esm --freeze_chem"

# Add WandB logging if requested
if $LOG_WANDB; then
    CMD="$CMD --log_wandb"
fi

# Print and execute command
echo "Running: $CMD"
eval $CMD