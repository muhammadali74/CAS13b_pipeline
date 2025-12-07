from pipeline import *

model = None
X_scaled = None
file_path = '/home/mauli/repos/CAS13b_pipeline/mlmodels/ml_output/feature_importance.csv'
with open(file_path) as f:
    feature_importance = pd.read_csv(f)
    feature_names = feature_importance['feature'].tolist()
def predict_new_spacers(model_path, spacers_csv, spacer_col='spacer'):
    """Predict efficiency for new crRNA spacer sequences."""
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    global model, X_scaled
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_extractor = model_data['feature_extractor']
    
    df = pd.read_csv(spacers_csv)
    df = df[df[spacer_col].str.len() == 30]
    
    X = feature_extractor.transform(df[spacer_col].tolist())
    X_scaled = scaler.transform(X)
    
    df['predicted_efficiency'] = model.predict(X_scaled)
    df['rank'] = df['predicted_efficiency'].rank(ascending=False)
    
    return df.sort_values('predicted_efficiency', ascending=False)

# Usage:
predictions = predict_new_spacers('/home/mauli/repos/CAS13b_pipeline/mlmodels/ml_output/cas13b_efficiency_model.pkl', '/home/mauli/repos/CAS13b_pipeline/per_subtype/spacers_conservation_filtered_optimized_subtypes.csv')

predictions.to_csv('/home/mauli/repos/CAS13b_pipeline/per_subtype/spacers_conservation_filtered_optimized_predicted.csv', index=False)
# predictions = predict_new_spacers('ml_output/cas13b_efficiency_model.pkl', '/home/mauli/repos/CAS13b_pipeline/mlmodels/Data_high_dosage.csv')

# predictions.to_csv('/home/mauli/repos/CAS13b_pipeline/mlmodels/Data_high_dosage_predicted.csv', index=False)


import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def perform_shap_analysis(model, X_scaled, feature_names, output_dir='ml_output'):
    """Perform SHAP analysis for model interpretability."""
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names,
                      max_display=30, show=False)
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature importance from SHAP
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_names[:len(mean_shap)],
        'mean_abs_shap': mean_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    shap_df.to_csv(f"{output_dir}/shap_importance.csv", index=False)
    return shap_values, shap_df

perform_shap_analysis(model, X_scaled, feature_names)