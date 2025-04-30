import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb
from tqdm import tqdm
import os
import sys
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score
)
# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import get_args
from utils.utils import set_seed, init_wandb, create_save_dir, log_metrics
from utils.data_utils import load_data, create_amino_acid_mapping, prepare_data_loaders
from model.models import BiTEAM

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    eval_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            esm_input = batch['esm_input']
            site = batch['site']
            chem_input = batch['chem_input']
            targets = batch['targets']
            
            logits = model(
                esm_input=esm_input,
                chem_input=chem_input,
                Site=site
            )
            
            loss = criterion(logits, targets)
            eval_loss += loss.item() * targets.size(0)
            
            _, predicted = torch.max(logits, 1)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of the positive class
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    eval_loss /= len(all_targets)
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    
    # Handle single-class scenarios for ROC AUC and MCC
    roc_auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
    mcc = matthews_corrcoef(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    metrics = {
        "loss": eval_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return metrics, all_probs

def train_epoch(model, data_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        esm_input = batch['esm_input']
        site = batch['site']
        chem_input = batch['chem_input']
        targets = batch['targets']
       
        optimizer.zero_grad()
        
        logits = model(
            esm_input=esm_input,
            chem_input=chem_input,
            Site=site
        )
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * targets.size(0)
        
        _, predicted = torch.max(logits, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return {"loss": epoch_loss, "accuracy": epoch_acc}

def train_model(args):
    """Main training function."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize WandB
    if args.log_wandb:
        config = init_wandb(args)
    
    # Create directory for saving models
    create_save_dir(args.save_dir)
    
    # Load data
    sequence_df, mapping_df, smiles_df = load_data(args.processed_data_path, args.nnaa_data_path)
    amino_acid_mapping = create_amino_acid_mapping(mapping_df)
    
    # Initialize model
    model = BiTEAM.from_pretrained(args, device)
    
    # Print trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    # Prepare data loaders
    train_loader, test_loader = prepare_data_loaders(
        args, amino_acid_mapping, sequence_df, smiles_df
    )
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    
    # Track best model
    best_roc_auc = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 20)
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        log_metrics(train_metrics, epoch, prefix="Train", log_wandb=args.log_wandb)
        
        # Evaluate model
        val_metrics, _ = evaluate_model(model, test_loader, criterion, device)
        log_metrics(val_metrics, epoch, prefix="Validation", log_wandb=args.log_wandb)
        # log_metrics(val_metrics, epoch, log_wandb=args.log_wandb)

        
        # Save model if ROC AUC is the best so far
        if val_metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = val_metrics["roc_auc"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model_seed_{args.seed}.pt'))
            print(f"Best ROC AUC improved to {best_roc_auc:.4f} at epoch {best_epoch}. Model checkpoint saved.")
        
    
    print(f"Training complete. Best ROC AUC: {best_roc_auc:.4f} at epoch {best_epoch}.")
    
    # Final evaluation
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'best_model_seed_{args.seed}.pt')))
    final_metrics, _ = evaluate_model(model, test_loader, criterion, device)
    print("\nFinal evaluation metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Close WandB
    if args.log_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = get_args()
    train_model(args)