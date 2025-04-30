import torch
import numpy as np
import random
import os
import wandb

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def init_wandb(args):
    """Initialize Weights and Biases logging."""
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "AdamW",
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "esm_model": args.esm_model,
                "chem_model": args.chem_model,
                "hidden_dim": args.hidden_dim,
                "prompt_dim": args.prompt_dim,
                "scheduler_step_size": args.scheduler_step_size,
                "scheduler_gamma": args.scheduler_gamma,
            }
        )
        return wandb.config
    return args

def create_save_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_metrics(metrics, epoch, prefix="", log_wandb=True):
    """Log metrics to console and optionally to WandB."""
    # Print metrics
    metrics_str = f"{prefix} Epoch {epoch+1}: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(metrics_str)
    
    # Log to WandB
    if log_wandb:
        wandb_metrics = {f"{prefix.lower()}_{k}": v for k, v in metrics.items()}
        wandb_metrics["epoch"] = epoch + 1
        wandb.log(wandb_metrics)