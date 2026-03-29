import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from som_core import (
    DATA_DIR, MODEL_DIR, MAPPED_DATA_PATH, BATCH_SIZE, DEVICE, 
    L1_GRID, L2_GRID, EPOCHS, DeepSOM
)

def load_data_for_inference(num_files=1):
    print("Loading data for inference...")
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))[:num_files]
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def run_inference():
    # 1. Load Preprocessor
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Preprocessor not found. Run train.py first.")
    
    pipeline = joblib.load(preprocessor_path)
    print("Preprocessor loaded.")

    # 2. Load and Transform Data
    df_raw = load_data_for_inference()
    X_raw = df_raw.drop(columns=['user_id'])
    X_processed = pipeline.transform(X_raw)
    x_tensor = torch.tensor(X_processed, dtype=torch.float32)

    # 3. Load Model
    model_path = os.path.join(MODEL_DIR, "deep_som.pth")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    input_dim = checkpoint['input_dim']
    
    model = DeepSOM(input_dim=input_dim, l1_grid=L1_GRID, l2_grid=L2_GRID, max_epochs=EPOCHS)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(DEVICE)
    print("Deep SOM model loaded.")

    # 4. Map Data to Segments
    print("Running inference batches...")
    l1_indices, l2_indices = [], []
    
    with torch.no_grad():
        dataloader = DataLoader(TensorDataset(x_tensor), batch_size=BATCH_SIZE)
        for (x_batch,) in dataloader:
            l1_idx, l2_idx = model(x_batch.to(DEVICE), 0)
            l1_indices.append(l1_idx.cpu().numpy())
            l2_indices.append(l2_idx.cpu().numpy())
            
    df_raw['segment_l1'] = np.concatenate(l1_indices)
    df_raw['segment_l2'] = np.concatenate(l2_indices)
    
    # 5. Save Segmented Data
    df_raw.to_parquet(MAPPED_DATA_PATH, engine='pyarrow', index=False)
    print(f"Inference complete. Mapped data saved to {MAPPED_DATA_PATH}")

if __name__ == "__main__":
    run_inference()