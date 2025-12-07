#!/usr/bin/env python3
"""
RNA-FM Based Deep Learning Pipeline for PspCas13b crRNA Efficiency Prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import os



class RNAFMEmbeddingExtractor:
    """Extract embeddings from RNA-FM foundation model."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
    def load_model(self):
        """Load the pretrained RNA-FM model."""
        import fm
        
        print("Loading RNA-FM pretrained model...")
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"RNA-FM loaded on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def extract_embeddings(self, sequences, layer=12, pooling='mean', batch_size=32):
        """
        Extract embeddings from RNA-FM.
        
        Parameters:
        -----------
        sequences : list of str
            RNA/DNA sequences (T will be converted to U)
        layer : int
            Transformer layer to extract from (1-12, default 12)
        pooling : str
            'mean', 'max', 'cls', or 'concat'
        
        Returns:
        --------
        np.ndarray : [N, 640] embedding matrix (or [N, 1280] for 'concat')
        """
        if self.model is None:
            self.load_model()
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
            batch_seqs = sequences[i:i+batch_size]
            
            # Prepare: convert T→U, create (name, seq) tuples
            data = [(f"seq_{j}", seq.upper().replace('T', 'U')) 
                    for j, seq in enumerate(batch_seqs)]
            
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[layer])
            
            token_emb = results["representations"][layer]  # [B, L, 640]
            
            for j, seq in enumerate(batch_strs):
                seq_len = len(seq)
                seq_emb = token_emb[j, 1:seq_len+1, :]  # exclude special tokens
                
                if pooling == 'mean':
                    pooled = seq_emb.mean(dim=0)
                elif pooling == 'max':
                    pooled = seq_emb.max(dim=0)[0]
                elif pooling == 'cls':
                    pooled = token_emb[j, 0, :]
                elif pooling == 'concat':
                    pooled = torch.cat([seq_emb.mean(dim=0), seq_emb.max(dim=0)[0]])
                else:
                    pooled = seq_emb.mean(dim=0)
                
                all_embeddings.append(pooled.cpu().numpy())
        
        return np.array(all_embeddings)


# =============================================================================
# PREDICTION HEAD
# =============================================================================

class MLPPredictionHead(nn.Module):
    """MLP prediction head for efficiency regression."""
    
    def __init__(self, input_dim=640, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)


# =============================================================================
# TRAINER
# =============================================================================

class CRISPREfficiencyTrainer:
    """Trainer for crRNA efficiency prediction using RNA-FM embeddings."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.extractor = RNAFMEmbeddingExtractor(device)
        self.model = None
        
    def prepare_data(self, df, spacer_col='spacer', target_col='efficiency'):
        """Extract embeddings and prepare labels."""
        df = df.dropna(subset=[spacer_col, target_col])
        df = df[df[spacer_col].str.len() == 30]
        
        print(f"Samples: {len(df)}")
        
        sequences = df[spacer_col].tolist()
        labels = df[target_col].values.astype(np.float32)
        
        embeddings = self.extractor.extract_embeddings(sequences, layer=12, pooling='mean')
        print(f"Embeddings shape: {embeddings.shape}")
        
        return embeddings, labels
    
    def cross_validate(self, embeddings, labels, n_folds=10, epochs=100, 
                       lr=1e-3, batch_size=32):
        """K-fold cross-validation."""
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_preds = np.zeros(len(labels))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(embeddings)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
            
            X_train = torch.FloatTensor(embeddings[train_idx])
            X_val = torch.FloatTensor(embeddings[val_idx])
            y_train = torch.FloatTensor(labels[train_idx])
            y_val = torch.FloatTensor(labels[val_idx])
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                       batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), 
                                     batch_size=batch_size)
            
            # Create model
            model = MLPPredictionHead(input_dim=640).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            criterion = nn.MSELoss()
            
            # Train with early stopping
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Train
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                
                # Validate
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        val_loss += criterion(model(xb), yb).item() * len(xb)
                val_loss /= len(val_loader.dataset)
                
                scheduler.step(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        break
            
            # Evaluate
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                preds = model(X_val.to(self.device)).cpu().numpy()
            
            all_preds[val_idx] = preds
            
            r2 = r2_score(labels[val_idx], preds)
            spear = spearmanr(labels[val_idx], preds)[0]
            fold_results.append({'r2': r2, 'spearman': spear})
            print(f"  R²: {r2:.4f}, Spearman: {spear:.4f}")
        
        # Summary
        print("\n" + "="*50)
        print("CROSS-VALIDATION RESULTS")
        print("="*50)
        print(f"R²: {np.mean([r['r2'] for r in fold_results]):.4f} ± "
              f"{np.std([r['r2'] for r in fold_results]):.4f}")
        print(f"Spearman: {np.mean([r['spearman'] for r in fold_results]):.4f} ± "
              f"{np.std([r['spearman'] for r in fold_results]):.4f}")
        print(f"Overall Spearman: {spearmanr(labels, all_preds)[0]:.4f}")
        
        return fold_results, all_preds
    
    def train_final_model(self, embeddings, labels, epochs=150, lr=1e-3, batch_size=32):
        """Train final model on all data."""
        X = torch.FloatTensor(embeddings)
        y = torch.FloatTensor(labels)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        
        self.model = MLPPredictionHead(input_dim=640).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
        
        print(f"Final model trained on {len(labels)} samples")
        return self.model
    
    def predict(self, sequences):
        """Predict efficiency for new sequences."""
        embeddings = self.extractor.extract_embeddings(sequences, layer=12, pooling='mean')
        
        self.model.eval()
        X = torch.FloatTensor(embeddings).to(self.device)
        with torch.no_grad():
            return self.model(X).cpu().numpy()
    
    def save_model(self, path='model.pt'):
        torch.save({'model_state': self.model.state_dict()}, path)
    
    def load_model(self, path='model.pt'):
        self.model = MLPPredictionHead(input_dim=640).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device)['model_state'])




def run_pipeline(csv_path, spacer_col='spacer', target_col='efficiency', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    trainer = CRISPREfficiencyTrainer()
    
    # Prepare data
    embeddings, labels = trainer.prepare_data(df, spacer_col, target_col)
    np.save(f"{output_dir}/embeddings.npy", embeddings)
    
    # Cross-validate
    results, preds = trainer.cross_validate(embeddings, labels, n_folds=10, epochs=100)
    
    # Train final model
    trainer.train_final_model(embeddings, labels, epochs=150)
    trainer.save_model(f"{output_dir}/model.pt")
    
    # Save predictions
    df['predicted'] = preds
    df.to_csv(f"{output_dir}/predictions.csv", index=False)
    
    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Input CSV')
    parser.add_argument('--spacer-col', default='spacer')
    parser.add_argument('--target-col', default='efficiency')
    parser.add_argument('--output-dir', default='rnafm_output')
    args = parser.parse_args()
    
    run_pipeline(args.csv, args.spacer_col, args.target_col, args.output_dir)