import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Peptide Classification Model")
    
    # Basic settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cuda:1, cpu, etc.)')
    
    # Data paths
    parser.add_argument('--processed_data_path', type=str, default='./data/pampa.csv', 
                        help='Path to the processed data CSV')
    parser.add_argument('--nnaa_data_path', type=str, default='./data/ncaa.xlsx', 
                        help='Path to the NNAAs Excel file')
    
    # Model parameters
    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t6_8M_UR50D', 
                        help='ESM model to use')
    parser.add_argument('--chem_model', type=str, default='ibm/materials.selfies-ted', 
                        help='SELFIES model to use')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for classification head')
    parser.add_argument('--prompt_dim', type=int, default=256, help='Prompt dimension')
    parser.add_argument('--freeze_esm', action='store_true', help='Whether to freeze ESM model weights')
    parser.add_argument('--freeze_chem', action='store_true', help='Whether to freeze chemistry model weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='Step size for LR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma for LR scheduler')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split size')
    
    # Logging parameters
    parser.add_argument('--wandb_project', type=str, help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, help='WandB entity name')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for WandB')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_wandb', action='store_true', help='Whether to log metrics to WandB')
    
    args = parser.parse_args()
    
    # Create run name if not provided
    if args.run_name is None:
        args.run_name = f'ours_pampa_3layers_relu(256->128)_seed_{args.seed}'
    
    return args