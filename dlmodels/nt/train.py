#!/usr/bin/env python3
"""Nucleotide Transformer Pipeline for crRNA Efficiency Prediction"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from tqdm import tqdm
import os


class NTEmbeddingExtractor:
    """Extract embeddings from Nucleotide Transformer."""
    
    def __init__(self, model_name="InstaDeepAI/nucleotide-transformer-500m-1000g",
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=False)
        self.model = self.model.to(self.device).eval()
        self.embedding_dim = self.model.config.hidden_size
        print(f"Loaded: {self.embedding_dim}-dim embeddings")
    
    def extract_embeddings(self, sequences, batch_size=32):
        if self.model is None:
            self.load_model()
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting"):
            batch = [s.upper().replace('U', 'T') for s in sequences[i:i+batch_size]]
            
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)
            
            hidden = outputs.hidden_states[-1]  # Last layer
            
            for j in range(len(batch)):
                mask = attention_mask[j].unsqueeze(-1)
                pooled = (hidden[j] * mask).sum(dim=0) / mask.sum()
                all_embeddings.append(pooled.cpu().numpy())
        
        return np.array(all_embeddings)


class MLPHead(nn.Module):
    """MLP prediction head."""
    
    def __init__(self, input_dim=1024, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class NTTrainer:
    """Trainer for Nucleotide Transformer + MLP."""
    
    def __init__(self, model_name="InstaDeepAI/nucleotide-transformer-500m-1000g"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = NTEmbeddingExtractor(model_name, self.device)
        self.model = None
    
    def prepare_data(self, df, spacer_col='spacer', target_col='efficiency'):
        df = df.dropna(subset=[spacer_col, target_col])
        df = df[df[spacer_col].str.len() == 30]
        
        embeddings = self.extractor.extract_embeddings(df[spacer_col].tolist())
        labels = df[target_col].values.astype(np.float32)
        
        print(f"Embeddings: {embeddings.shape}")
        return embeddings, labels
    
    def cross_validate(self, embeddings, labels, n_folds=10, epochs=100):
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []
        all_preds = np.zeros(len(labels))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(embeddings)):
            print(f"\nFold {fold+1}/{n_folds}")
            
            X_train = torch.FloatTensor(embeddings[train_idx])
            X_val = torch.FloatTensor(embeddings[val_idx])
            y_train = torch.FloatTensor(labels[train_idx])
            y_val = torch.FloatTensor(labels[val_idx])
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
            
            model = MLPHead(input_dim=embeddings.shape[1]).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.MSELoss()
            
            best_loss = float('inf')
            patience = 0
            
            for epoch in range(epochs):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    criterion(model(xb), yb).backward()
                    optimizer.step()
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        val_loss += criterion(model(xb), yb).item() * len(xb)
                val_loss /= len(val_loader.dataset)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = model.state_dict().copy()
                    patience = 0
                else:
                    patience += 1
                    if patience >= 15:
                        break
            
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                preds = model(X_val.to(self.device)).cpu().numpy()
            
            all_preds[val_idx] = preds
            r2 = r2_score(labels[val_idx], preds)
            spear = spearmanr(labels[val_idx], preds)[0]
            results.append({'r2': r2, 'spearman': spear})
            print(f"  R²: {r2:.4f}, Spearman: {spear:.4f}")
        
        print(f"\nMean R²: {np.mean([r['r2'] for r in results]):.4f}")
        print(f"Mean Spearman: {np.mean([r['spearman'] for r in results]):.4f}")
        print(f"Overall Spearman: {spearmanr(labels, all_preds)[0]:.4f}")
        
        return results, all_preds
    
    def train_final(self, embeddings, labels, epochs=150):
        X = torch.FloatTensor(embeddings)
        y = torch.FloatTensor(labels)
        loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
        
        self.model = MLPHead(input_dim=embeddings.shape[1]).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(xb), yb).backward()
                optimizer.step()
        
        return self.model
    
    def predict(self, sequences):
        emb = self.extractor.extract_embeddings(sequences)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(emb).to(self.device)).cpu().numpy()
    
    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'dim': self.extractor.embedding_dim
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model = MLPHead(input_dim=ckpt['dim']).to(self.device)
        self.model.load_state_dict(ckpt['model'])


def run_pipeline(csv_path, spacer_col='spacer', target_col='efficiency', output_dir='nt_output'):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    trainer = NTTrainer()
    
    embeddings, labels = trainer.prepare_data(df, spacer_col, target_col)
    np.save(f"{output_dir}/embeddings.npy", embeddings)
    
    results, preds = trainer.cross_validate(embeddings, labels)
    
    trainer.train_final(embeddings, labels)
    trainer.save(f"{output_dir}/model.pt")
    
    df['predicted'] = preds
    df.to_csv(f"{output_dir}/predictions.csv", index=False)
    
    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--spacer-col', default='spacer')
    parser.add_argument('--target-col', default='efficiency')
    parser.add_argument('--output-dir', default='nt_output')
    args = parser.parse_args()
    
    run_pipeline(args.csv, args.spacer_col, args.target_col, args.output_dir)