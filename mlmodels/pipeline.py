import pandas as pd
import numpy as np
from sklearn.model_selection import (
    KFold, cross_val_score, cross_val_predict,
    GridSearchCV, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from collections import Counter
from itertools import product
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class SpacerFeatureExtractor:
    """Extract features from 30nt crRNA spacer sequences."""
    
    def __init__(self, spacer_length=30):
        self.spacer_length = spacer_length
        self.nucleotides = ['A', 'C', 'G', 'T']
        self.feature_names = []
        
    def _validate_sequence(self, seq):
        """Validate and clean sequence."""
        seq = seq.upper().replace('U', 'T')
        if len(seq) != self.spacer_length:
            raise ValueError(f"Sequence length {len(seq)} != {self.spacer_length}")
        return seq
    
    def one_hot_encode(self, seq):
        """One-hot encode a sequence. Returns: 120 features"""
        seq = self._validate_sequence(seq)
        encoding = []
        for pos, nuc in enumerate(seq):
            for n in self.nucleotides:
                encoding.append(1 if nuc == n else 0)
        return encoding
    
    def get_kmer_frequencies(self, seq, k):
        """Get normalized k-mer frequencies."""
        seq = self._validate_sequence(seq)
        kmers = [''.join(p) for p in product(self.nucleotides, repeat=k)]
        counts = Counter([seq[i:i+k] for i in range(len(seq) - k + 1)])
        total = sum(counts.values())
        return [counts.get(kmer, 0) / total if total > 0 else 0 for kmer in kmers]
    
    def gc_content(self, seq):
        """Calculate overall GC content."""
        seq = self._validate_sequence(seq)
        return sum(1 for n in seq if n in 'GC') / len(seq)
    
    def windowed_gc_content(self, seq, window_size=10):
        """Calculate GC content in sliding windows."""
        seq = self._validate_sequence(seq)
        gc_values = []
        for i in range(0, len(seq) - window_size + 1, window_size // 2):
            window = seq[i:i + window_size]
            gc_values.append(sum(1 for n in window if n in 'GC') / len(window))
        return gc_values
    
    def melting_temperature(self, seq):
        """Approximate melting temperature."""
        seq = self._validate_sequence(seq)
        g_count = seq.count('G')
        c_count = seq.count('C')
        return 64.9 + 41 * (g_count + c_count - 16.4) / len(seq)
    
    def sequence_complexity(self, seq):
        """Calculate sequence complexity metrics."""
        seq = self._validate_sequence(seq)
        freq = Counter(seq)
        probs = [count / len(seq) for count in freq.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        dinucs = [seq[i:i+2] for i in range(len(seq) - 1)]
        dinuc_freq = Counter(dinucs)
        max_dinuc_freq = max(dinuc_freq.values()) / len(dinucs)
        return [entropy, max_dinuc_freq]
    
    def positional_gc(self, seq):
        """GC content at different regions."""
        seq = self._validate_sequence(seq)
        third = len(seq) // 3
        regions = [seq[:third], seq[third:2*third], seq[2*third:]]
        return [sum(1 for n in r if n in 'GC') / len(r) for r in regions]
    
    def nucleotide_counts(self, seq):
        """Simple nucleotide counts."""
        seq = self._validate_sequence(seq)
        return [seq.count(n) for n in self.nucleotides]
    
    def purine_pyrimidine_ratio(self, seq):
        """Calculate purine to pyrimidine ratio."""
        seq = self._validate_sequence(seq)
        purines = sum(1 for n in seq if n in 'AG')
        pyrimidines = sum(1 for n in seq if n in 'CT')
        return purines / pyrimidines if pyrimidines > 0 else 0
    
    def extract_all_features(self, seq):
        """Extract all features from a single sequence."""
        features = []
        features.extend(self.one_hot_encode(seq))           # 120 features
        features.extend(self.get_kmer_frequencies(seq, 2))  # 16 features
        features.extend(self.get_kmer_frequencies(seq, 3))  # 64 features
        features.append(self.gc_content(seq))               # 1 feature
        features.extend(self.positional_gc(seq))            # 3 features
        features.extend(self.windowed_gc_content(seq, 10))  # ~5 features
        features.extend(self.nucleotide_counts(seq))        # 4 features
        features.append(self.melting_temperature(seq))      # 1 feature
        features.extend(self.sequence_complexity(seq))      # 2 features
        features.append(self.purine_pyrimidine_ratio(seq))  # 1 feature
        return features
    
    def get_feature_names(self):
        """Generate feature names."""
        names = []
        for pos in range(1, self.spacer_length + 1):
            for nuc in self.nucleotides:
                names.append(f"{nuc}_pos{pos}")
        for di in [''.join(p) for p in product(self.nucleotides, repeat=2)]:
            names.append(f"dinuc_{di}")
        for tri in [''.join(p) for p in product(self.nucleotides, repeat=3)]:
            names.append(f"trinuc_{tri}")
        names.append("gc_content_overall")
        names.extend(["gc_5prime", "gc_middle", "gc_3prime"])
        names.extend([f"gc_window_{i}" for i in range(5)])
        names.extend([f"count_{n}" for n in self.nucleotides])
        names.append("melting_temp")
        names.extend(["shannon_entropy", "max_dinuc_repeat"])
        names.append("purine_pyrimidine_ratio")
        self.feature_names = names
        return names
    
    def transform(self, sequences):
        """Transform a list of sequences to feature matrix."""
        return np.array([self.extract_all_features(seq) for seq in sequences])


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

class CRISPREfficiencyPredictor:
    """Train and evaluate ML models for crRNA efficiency prediction."""
    
    def __init__(self, n_folds=10, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.feature_extractor = SpacerFeatureExtractor()
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        
    def prepare_data(self, df, spacer_col='spacer', target_col='efficiency'):
        """Prepare data for training."""
        print("="*70)
        print("PREPARING DATA")
        print("="*70)
        
        df = df.dropna(subset=[spacer_col, target_col])
        df = df[df[spacer_col].str.len() == 30]
        
        print(f"Total samples: {len(df)}")
        print(f"Target range: {df[target_col].min():.3f} - {df[target_col].max():.3f}")
        print(f"Target mean: {df[target_col].mean():.3f}")
        
        sequences = df[spacer_col].tolist()
        X = self.feature_extractor.transform(sequences)
        y = df[target_col].values
        
        self.feature_names = self.feature_extractor.get_feature_names()
        if len(self.feature_names) != X.shape[1]:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        print(f"Feature matrix shape: {X.shape}")
        return X, y
    
    def get_models(self):
        """Define models to compare."""
        return {
            'XGBoost': XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=self.random_state, n_jobs=-1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10,
                min_samples_split=5, min_samples_leaf=2,
                random_state=self.random_state, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=self.random_state
            ),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        }
    
    def cross_validate_models(self, X, y):
        """Perform k-fold cross-validation for all models."""
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION ({self.n_folds}-FOLD)")
        print("="*70)
        
        X_scaled = self.scaler.fit_transform(X)
        models = self.get_models()
        results = {}
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            r2_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
            mse_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error')
            mae_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_mean_absolute_error')
            y_pred = cross_val_predict(model, X_scaled, y, cv=kfold)
            spearman = np.corrcoef(y, y_pred)[0, 1]
            
            results[name] = {
                'r2_mean': r2_scores.mean(), 'r2_std': r2_scores.std(),
                'rmse_mean': np.sqrt(-mse_scores.mean()),
                'mae_mean': -mae_scores.mean(),
                'spearman': spearman, 'predictions': y_pred
            }
            
            print(f"  R²: {results[name]['r2_mean']:.4f} ± {results[name]['r2_std']:.4f}")
            print(f"  RMSE: {results[name]['rmse_mean']:.4f}")
            print(f"  Spearman: {results[name]['spearman']:.4f}")
        
        return results, X_scaled
    
    def train_final_model(self, X, y, model_name='XGBoost', params=None):
        """Train the final model on all data."""
        print(f"\n{'='*70}")
        print(f"TRAINING FINAL MODEL ({model_name})")
        print("="*70)
        
        X_scaled = self.scaler.fit_transform(X)
        
        if model_name == 'XGBoost':
            model = XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=self.random_state, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=self.random_state, n_jobs=-1
            )
        
        model.fit(X_scaled, y)
        self.best_model = model
        print(f"Model trained on {len(y)} samples")
        return model
    
    def get_feature_importance(self, model=None, top_n=30):
        """Get feature importance."""
        if model is None:
            model = self.best_model
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop {top_n} important features:")
            print(importance_df.head(top_n).to_string(index=False))
            return importance_df
        return None
    
    def predict(self, sequences):
        """Predict efficiency for new sequences."""
        X = self.feature_extractor.transform(sequences)
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def save_model(self, filepath='cas13b_model.pkl'):
        """Save the trained model."""
        model_data = {
            'model': self.best_model, 'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_extractor': self.feature_extractor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='cas13b_model.pkl'):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_extractor = model_data['feature_extractor']


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(csv_path, spacer_col='spacer', target_col='efficiency',
                 output_dir='ml_output'):
    """Run the complete ML pipeline."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("CRISPR EFFICIENCY PREDICTION PIPELINE")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} samples")
    
    predictor = CRISPREfficiencyPredictor(n_folds=10, random_state=42)
    X, y = predictor.prepare_data(df, spacer_col=spacer_col, target_col=target_col)
    results, X_scaled = predictor.cross_validate_models(X, y)
    
    best_model_name = max(results, key=lambda x: results[x]['r2_mean'])
    print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['r2_mean']:.4f})")
    
    final_model = predictor.train_final_model(X, y, model_name=best_model_name)
    importance_df = predictor.get_feature_importance(final_model)
    
    if importance_df is not None:
        importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    predictor.save_model(f"{output_dir}/cas13b_efficiency_model.pkl")
    
    results_df = pd.DataFrame({
        k: {m: v for m, v in val.items() if m != 'predictions'}
        for k, val in results.items()
    }).T
    results_df.to_csv(f"{output_dir}/model_comparison.csv")
    
    print(f"\nOutputs saved to: {output_dir}/")
    return predictor, results, importance_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', help='Input CSV')
    parser.add_argument('--spacer-col', default='spacer')
    parser.add_argument('--target-col', default='efficiency')
    parser.add_argument('--output-dir', default='ml_output')
    
    args = parser.parse_args()
    run_pipeline(args.input_csv, args.spacer_col, args.target_col, args.output_dir)