import os
import glob
import time
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import shared architecture
from som_core import (
    DATA_DIR, MODEL_DIR, BATCH_SIZE, DEVICE, L1_GRID, L2_GRID, EPOCHS,
    DataPipeline, DeepSOM
)

def load_training_data(pipeline, num_files=2):
    print("Loading data for training...")
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))[:num_files]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}.")
    
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    print("Fitting preprocessing pipeline...")
    X_raw = df.drop(columns=['user_id'])
    X_processed = pipeline.fit_transform(X_raw)
    
    return torch.tensor(X_processed, dtype=torch.float32)

def train():
    pipeline = DataPipeline()
    X_tensor = load_training_data(pipeline, num_files=2)
    input_dim = X_tensor.shape[1]
    
    # Save the fitted preprocessor so inference applies the exact same scaling
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    print("Preprocessor saved.")

    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Initializing Deep SOM on {DEVICE}. Feature dim: {input_dim}")
    model = DeepSOM(input_dim=input_dim, l1_grid=L1_GRID, l2_grid=L2_GRID, max_epochs=EPOCHS)
    
    model.train()
    model.to(DEVICE)
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        for batch_idx, (x_batch,) in enumerate(dataloader):
            x_batch = x_batch.to(DEVICE)
            model(x_batch, epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")
        
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Save Model Weights and configuration
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': input_dim
    }, os.path.join(MODEL_DIR, "deep_som.pth"))
    print("Model saved successfully.")

if __name__ == "__main__":
    train()