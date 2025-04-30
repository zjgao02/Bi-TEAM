import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import selfies as sf
import argparse
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.config import get_args
from utils.utils import set_seed
from utils.data_utils import create_amino_acid_mapping
from model.models import BiTEAM

amino_acid_map = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
    "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

def parse_inference_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="BiTEAM Model Inference")
    
    # Basic settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    
    # Model parameters
    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t6_8M_UR50D', 
                        help='ESM model to use')
    parser.add_argument('--chem_model', type=str, default='ibm/materials.selfies-ted', 
                        help='Chemistry model to use')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for classification head')
    parser.add_argument('--prompt_dim', type=int, default=256, help='Prompt dimension')
    parser.add_argument('--freeze_esm', action='store_true', help='Whether to freeze ESM model weights')
    parser.add_argument('--freeze_chem', action='store_true', help='Whether to freeze chemistry model weights')
    
    # Data paths
    parser.add_argument('--input_data_path', type=str, 
                        help='Path to input data CSV')
    parser.add_argument('--mapping_data_path', type=str,
                        help='Path to amino acid mapping file')
    parser.add_argument('--model_checkpoint', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save results')
    parser.add_argument('--output_name', type=str, 
                        help='Name of output file')
    
    args = parser.parse_args()
    return args

class InferenceDataset(torch.utils.data.Dataset):
    """Dataset for inference."""
    def __init__(self, df, amino_acid_mapping,mapping_df):
        self.df = df
        self.amino_acid_mapping = amino_acid_mapping
        self.indexes = df.index.tolist()
        smiles_map = dict(zip(mapping_df['Symbol'],mapping_df['replaced_SMILES']))
        def convert_to_single_letter(sequence):
            return ''.join([amino_acid_map.get(aa,aa) for aa in sequence.split('-')])

        self.smiles = []
        for seq in df['Sequence']:
            smiles_sequence = ''.join([smiles_map.get(aa,'') for aa in seq.split('-')])
            self.smiles.append(smiles_sequence)
        # Extract necessary columns
        self.sequences = df['Sequence'].apply(convert_to_single_letter).values
        # self.smiles = df['SMILES'].values
        self.permeability_class = df['Permeability_class'].values if 'Permeability_class' in df.columns else [-1] * len(df)
        
        # Extract additional columns if they exist
        self.helm = df['HELM'].values if 'HELM' in df.columns else [''] * len(df)
        self.pampa = df['PAMPA'].values if 'PAMPA' in df.columns else [0.0] * len(df)
        self.source = df['Source'].values if 'Source' in df.columns else [''] * len(df)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Map sequence to analogs and site vector
        analogs, site = self.map_sequence(sequence)
        concat_seq = ''.join([aa for aa in analogs])
        
        # Encode SMILES to SELFIES
        smiles = self.smiles[idx]
        chem_input = sf.encoder(smiles).replace("][", "] [")
        
        # Prepare target (if available)
        try:
            target = int(self.permeability_class[idx])
        except:
            target = -1  # Use -1 for unknown targets
        
        return {
            'esm_input': concat_seq,
            'chem_input': chem_input,
            'site': site,
            'targets': target,
            'index': self.indexes[idx],
            'source': self.source[idx],
            'smiles': smiles,
            'helm': self.helm[idx],
            'sequence': sequence,
            'pampa': self.pampa[idx],
            'permeability_class': target
        }
    
    def map_sequence(self, sequence):
        """Maps a peptide sequence to its amino acid analogs and site vector."""
        sequence_list = list(sequence)
        analogs = []
        site = []
        
        for aa in sequence_list:
            mapping = self.amino_acid_mapping.get(aa, {'analog': 'X', 'type': 'unnatural'})
            analogs.append(mapping['analog'])
            site.append(1 if mapping['type'] == 'unnatural' else 0)
        
        return analogs, site

def collate_fn(batch, device):
    """Custom collate function for inference batch processing."""
    # Extract all fields from the batch
    esm_inputs = [item['esm_input'] for item in batch]
    chem_inputs = [item['chem_input'] for item in batch]
    sites = [item['site'] for item in batch]
    targets = [item['targets'] for item in batch]
    indexes = [item['index'] for item in batch]
    sources = [item['source'] for item in batch]
    smiles = [item['smiles'] for item in batch]
    helms = [item['helm'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    pampas = [item['pampa'] for item in batch]
    permeability_classes = [item['permeability_class'] for item in batch]

    # Find the maximum sequence length for 'site' padding
    max_len = max([len(site) for site in sites])
    
    # Pad 'site' to match the max sequence length in the batch
    padded_sites = []
    for site in sites:
        site_tensor = torch.tensor(site, dtype=torch.long)
        if len(site) < max_len:
            padding = torch.zeros(max_len - len(site), dtype=torch.long)
            site_tensor = torch.cat([site_tensor, padding], dim=0)
        padded_sites.append(site_tensor)
    
    padded_sites = torch.stack(padded_sites, dim=0).to(device)
    targets = torch.tensor(targets, dtype=torch.long).to(device)

    return {
        'esm_input': esm_inputs,
        'chem_input': chem_inputs,
        'site': padded_sites,
        'targets': targets,
        'index': indexes,
        'source': sources,
        'smiles': smiles,
        'helm': helms,
        'sequence': sequences,
        'pampa': pampas,
        'permeability_class': permeability_classes
    }

def evaluate_and_save_results(model, data_loader, args):
    """Run inference and save results to file."""
    model.eval()
    results = []
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running inference"):
            esm_input = batch['esm_input']
            site = batch['site']
            chem_input = batch['chem_input']
            targets = batch['targets']
            

            # Additional metadata
            indexes = batch['index']
            sources = batch['source']
            smiles = batch['smiles']
            helms = batch['helm']
            sequences = batch['sequence']
            pampas = batch['pampa']
            permeability_classes = batch['permeability_class']
            
            # Forward pass
            
            logits = model(
                esm_input=esm_input,
                chem_input=chem_input,
                Site=site
            )
            
            # Calculate probabilities and predictions
            probs = torch.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            _, predicted = torch.max(logits, 1)
            
            # Collect predictions and targets for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend([t for t in targets.cpu().numpy() if t != -1])  # Only include known targets
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            # Store results for each sample
            for i in range(len(targets)):
                results.append({
                    'Index': indexes[i],
                    'Source': sources[i],
                    'SMILES': smiles[i],
                    'HELM': helms[i],
                    'Sequence': sequences[i],
                    'PAMPA': pampas[i],
                    'Permeability_class': permeability_classes[i],
                    'Prob_Positive_Class': probs[i, 1].item(),
                    'Predicted_Label': predicted[i].item(),
                    'True_Label': targets[i].item() if targets[i].item() != -1 else 'Unknown',
                    'Entropy': entropy[i].item()
                })
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = os.path.join(args.output_dir, args.output_name)
    results_df.to_csv(output_path, index=False)
    
    # Calculate and print metrics if we have ground truth labels
    if len(all_targets) > 0:
        metrics = calculate_metrics(all_targets, all_preds, all_probs)
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return results_df

def calculate_metrics(targets, predictions, probabilities):
    """Calculate performance metrics."""
    # Handle single class scenario
    if len(set(targets)) <= 1:
        print("Warning: Only one class present in targets, some metrics cannot be calculated.")
        return {
            "accuracy": (torch.tensor(predictions) == torch.tensor(targets)).float().mean().item(),
            "balanced_accuracy": 0.0,
            "roc_auc": 0.0,
            "mcc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    # Calculate metrics
    accuracy = (torch.tensor(predictions) == torch.tensor(targets)).float().mean().item()
    balanced_acc = balanced_accuracy_score(targets, predictions)
    roc_auc = roc_auc_score(targets, probabilities)
    mcc = matthews_corrcoef(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def main():
    """Main function for inference."""
    # Parse arguments
    args = parse_inference_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    input_df = pd.read_csv(args.input_data_path)
    mapping_df = pd.read_excel(args.mapping_data_path)
    
    # Create amino acid mapping
    amino_acid_mapping = {}
    for idx, row in mapping_df.iterrows():
        symbol = row['Symbol']
        natural_analog = row['Natural_Analog']
        if symbol == natural_analog:
            amino_acid_mapping[symbol] = {'analog': natural_analog, 'type': 'natural'}
        else:
            amino_acid_mapping[symbol] = {'analog': natural_analog, 'type': 'unnatural'}
    amino_acid_mapping['-pip'] = {'analog': 'X', 'type': 'unnatural'}
    
    # Create dataset and dataloader
    inference_dataset = InferenceDataset(input_df, amino_acid_mapping,mapping_df)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, device)
    )
    
    # Initialize model
    model = BiTEAM.from_pretrained(args, device)
    
    # Load checkpoint
    print(f"Loading model checkpoint from {args.model_checkpoint}")
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=True)
    model.eval()
    
    # Run inference and save results
    results_df = evaluate_and_save_results(model, inference_loader, args)
    
    
if __name__ == "__main__":
    main()