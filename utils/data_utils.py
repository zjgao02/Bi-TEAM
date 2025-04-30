import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import selfies as sf
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(processed_data_path, nnaa_data_path):
    """Load and preprocess data."""
    sequence_df = pd.read_csv(processed_data_path)
    mapping_df = pd.read_excel(nnaa_data_path)
    smiles_df = pd.read_excel(nnaa_data_path)
    
    return sequence_df, mapping_df, smiles_df

def create_amino_acid_mapping(mapping_df):
    """Create mapping dictionary for amino acids."""
    amino_acid_mapping = {}
    for idx, row in mapping_df.iterrows():
        symbol = row['Symbol']
        natural_analog = row['Natural_Analog']
        if symbol == natural_analog:
            amino_acid_mapping[symbol] = {'analog': natural_analog, 'type': 'natural'}
        else:
            amino_acid_mapping[symbol] = {'analog': natural_analog, 'type': 'unnatural'}
    amino_acid_mapping['-pip'] = {'analog': 'X', 'type': 'unnatural'}
    
    return amino_acid_mapping

def map_sequence(sequence, amino_acid_mapping):
    """
    Maps a peptide sequence to its amino acid analogs and types,
    and returns a Site vector where 1 indicates non-natural amino acids, and 0 otherwise.
    """
    sequence_list = eval(sequence)
    analogs = []
    site = []
   
    for aa in sequence_list:
        mapping = amino_acid_mapping.get(aa, {'analog': aa, 'type': 'natural'})
        analogs.append(mapping['analog'])
        site.append(1 if mapping['type'] == 'unnatural' else 0)
   
    return analogs, site

class CustomClassificationDataset(Dataset):
    def __init__(self, df, mapping, smiles_df, label_col='Permeability_class', device='cuda'):
        """
        Custom dataset for peptide classification.
        """
        self.sequences = []
        self.sites = []
        for seq in df['Sequence']:
            analogs, site = map_sequence(seq, mapping)
            self.sequences.append(analogs)
            self.sites.append(site)

        self.smiles = df['SMILES'].values
        self.targets = df[label_col].values
        self.smiles_map = dict(zip(smiles_df['Symbol'], smiles_df['replaced_SMILES']))
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        For each sample, returns ESM inputs, chemistry inputs, the Site vector, and the target label.
        """
        sequence = self.sequences[idx]
        site = self.sites[idx]
        smiles = self.smiles[idx]
        target = torch.tensor(int(self.targets[idx]), dtype=torch.long)
        
        # Concatenate amino acid symbols to form the sequence string for ESM
        concat_seq = ''.join([aa for aa in sequence])
        
        # Convert SMILES to SELFIES
        chem_input = sf.encoder(smiles).replace("][", "] [")
        
        return {
            'esm_input': concat_seq,
            'chem_input': chem_input,
            'site': site,
            'targets': target
        }

def collate_fn(batch, device):
    """
    Custom collate function for the DataLoader.
    """
    esm_inputs = [item['esm_input'] for item in batch]
    chem_inputs = [item['chem_input'] for item in batch]
    sites = [item['site'] for item in batch]
    targets = [item['targets'] for item in batch]
    
    # Find the batch's maximum sequence length
    max_len = max([len(site) for site in sites])
    
    # Padding Site to batch's maximum sequence length
    padded_sites = []
    for site in sites:
        site_tensor = torch.tensor(site, dtype=torch.long)
        if len(site) < max_len:
            padding = torch.zeros(max_len - len(site), dtype=torch.long)
            site_tensor = torch.cat([site_tensor, padding], dim=0)
        padded_sites.append(site_tensor)
    
    padded_sites = torch.stack(padded_sites, dim=0).to(device)
    targets = torch.stack(targets, dim=0).to(device)

    return {
        'esm_input': esm_inputs,
        'chem_input': chem_inputs,
        'site': padded_sites,
        'targets': targets
    }

def prepare_data_loaders(args, amino_acid_mapping, sequence_df, smiles_df):
    """Prepare training and testing data loaders."""
    train_df, test_df = train_test_split(
        sequence_df, 
        test_size=args.test_size, 
        random_state=args.seed
    )
    
    # Create Datasets
    train_dataset = CustomClassificationDataset(
        df=train_df,
        mapping=amino_acid_mapping,
        smiles_df=smiles_df,
        label_col='Permeability_class',
        device=args.device
    )

    test_dataset = CustomClassificationDataset(
        df=test_df,
        mapping=amino_acid_mapping,
        smiles_df=smiles_df,
        label_col='Permeability_class',
        device=args.device
    )

    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, args.device)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, args.device)
    )
    
    return train_loader, test_loader