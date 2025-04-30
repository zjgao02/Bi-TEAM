import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import selfies as sf

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        """
        Classification head with multi-layer architecture.
        """
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, num_classes)

    def forward(self, x):
        """
        Forward pass through the classification head.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
class ESMModule(nn.Module):
    """ESM language model module for sequence processing."""
    def __init__(self, model_name, freeze=True):
        super(ESMModule, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        # Freeze parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, sequences, device):
        # Process sequences with tokenizer
        batch_max_len = max([len(s) for s in sequences])
        inputs = self.tokenizer(
            sequences, 
            return_tensors='pt', 
            add_special_tokens=True, 
            truncation=True, 
            padding='max_length', 
            max_length=batch_max_len
        ).to(device)
        
        # Get model outputs
        outputs = self.model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            return_dict=True
        )
        
        # Return CLS token embeddings
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]


class ChemistryModule(nn.Module):
    """Chemistry module for processing chemical space representations."""
    def __init__(self, model_name, freeze=True):
        super(ChemistryModule, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        # Freeze parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, selfies_inputs, device):
        # Process SELFIES with tokenizer
        tokenized_inputs = self.tokenizer(
            selfies_inputs, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        ).to(device)
        
        # Get model outputs
        outputs = self.model.encoder(
            input_ids=tokenized_inputs['input_ids'], 
            attention_mask=tokenized_inputs['attention_mask']
        )
        hidden_states = outputs.last_hidden_state
        
        # Compute weighted average over sequence dimension
        attention_mask = tokenized_inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        avg_embeddings = sum_embeddings / sum_mask
        
        
        return avg_embeddings  # [batch_size, hidden_size] or [batch_size, output_dim]


class PromptModule(nn.Module):
    """Module for processing modify site information as prompts."""
    def __init__(self, d_prompt=256, max_length=4000):
        super(PromptModule, self).__init__()
        self.embedding = nn.Embedding(2, d_prompt)  # 0 for natural, 1 for unnatural
        self.positional_encodings = nn.Parameter(torch.rand(max_length, d_prompt), requires_grad=False)
        
        # Transformer encoder for prompt processing
        encoder_layer = nn.TransformerEncoderLayer(d_prompt, nhead=8, dropout=0.1)
        encoder_norm = nn.LayerNorm(d_prompt)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=encoder_norm)
        
        self.hidden_size = d_prompt
    
    def forward(self, site_tensor):
        # Embed site information
        site_embeddings = self.embedding(site_tensor)  # [batch_size, seq_len, d_prompt]
        
        # Add positional encodings
        site_embeddings = site_embeddings + self.positional_encodings[:site_embeddings.shape[1], :]
        
        # Process through transformer
        # Change from [batch, seq, features] to [seq, batch, features] for transformer
        site_embeddings = site_embeddings.permute(1, 0, 2)
        transformed_embeddings = self.transformer(site_embeddings)
        
        # Change back to [batch, seq, features]
        transformed_embeddings = transformed_embeddings.permute(1, 0, 2)
        
        # Average over sequence dimension
        avg_embeddings = torch.mean(transformed_embeddings, dim=1)
        
        return avg_embeddings  # [batch_size, d_prompt]


class FusionModule(nn.Module):
    """Module for fusing embeddings from different modalities."""
    def __init__(self, esm_dim, chem_dim, prompt_dim, output_dim=None):
        super(FusionModule, self).__init__()
        self.combined_dim = esm_dim + chem_dim + prompt_dim
        self.output_dim = esm_dim

        # Optional projection to match other embedding dimensions
        self.output_dim = output_dim
        if output_dim is not None:
            self.projection = nn.Linear(chem_dim, output_dim)
        else:
            self.projection = None
            
        # MLPs for different fusion strategies
        self.mlp_e = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Linear(self.combined_dim, esm_dim)
        )
        
        self.mlp_c = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Linear(self.combined_dim, esm_dim)
        )
        
        self.mlp_r = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Linear(self.combined_dim, esm_dim)
        )
        
        # Learnable weights for fusion
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
    
    def forward(self, esm_embeddings, chem_embeddings, prompt_embeddings):
        # Combine embeddings
        combined = torch.cat((esm_embeddings, chem_embeddings, prompt_embeddings), dim=-1)

        # Process through MLPs
        mlp_e_output = self.mlp_e(combined)
        mlp_c_output = self.mlp_c(combined)
        mlp_r_output = self.mlp_r(combined)

        # Apply chem projection if needed
        if self.projection is not None:
            chem_embeddings = self.projection(chem_embeddings)
            
        # Weighted fusion strategy
        weighted_output = self.weight1 * mlp_e_output * esm_embeddings + self.weight2 * mlp_c_output * chem_embeddings + mlp_r_output
        
        return weighted_output  # [batch_size, esm_dim]


class ClassificationHead(nn.Module):
    """Classification head for the final prediction."""
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # [batch_size, num_classes]


class BiTEAM(nn.Module):
    """
    Combines protein language model (ESM) with chemistry model (SELFIES) and prompt-based
    site information for unnatural amino acids.
    """
    def __init__(self, esm_module, chem_module, prompt_module, fusion_module, classification_head):
        super(BiTEAM, self).__init__()
        self.esm = esm_module
        self.chem = chem_module
        self.prompt = prompt_module
        self.fusion = fusion_module
        self.classifier = classification_head
    
    def forward(self, esm_input, chem_input, Site):
        # Get embeddings from each module
        esm_embeddings = self.esm(esm_input, Site.device)
        chem_embeddings = self.chem(chem_input, Site.device)
        prompt_embeddings = self.prompt(Site)
        
        # Fuse embeddings
        fused_embeddings = self.fusion(esm_embeddings, chem_embeddings, prompt_embeddings)
        
        # Get final prediction
        logits = self.classifier(fused_embeddings)
        
        return logits
    
    @classmethod
    def from_pretrained(cls, args, device):
        """Factory method to create BiTEAM model from pretrained components."""
        # Initialize ESM module
        esm_module = ESMModule(args.esm_model, freeze=args.freeze_esm)
        
        # Initialize Chemistry module
        chem_module = ChemistryModule(
            args.chem_model, 
            freeze=args.freeze_chem
        )
        
        # Initialize Prompt module
        prompt_module = PromptModule(d_prompt=args.prompt_dim)
        
        # Initialize Fusion module
        fusion_module = FusionModule(
            esm_dim=esm_module.hidden_size,
            chem_dim=chem_module.hidden_size,
            prompt_dim=prompt_module.hidden_size,
            output_dim=esm_module.hidden_size,
        )
        
        # Initialize Classification head
        classification_head = ClassificationHead(
            input_dim=esm_module.hidden_size,
            hidden_dim=args.hidden_dim,
            num_classes=2
        )
        
        # Create the complete model
        model = cls(
            esm_module=esm_module,
            chem_module=chem_module,
            prompt_module=prompt_module,
            fusion_module=fusion_module,
            classification_head=classification_head
        ).to(device)
        
        return model
    
