import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from som_core import (
    MODEL_DIR, VIS_DIR, MAPPED_DATA_PATH, DEVICE, 
    L1_GRID, L2_GRID, EPOCHS, DeepSOM
)

class SOMVisualizer:
    @staticmethod
    def plot_u_matrix(som_layer, layer_name):
        weights = som_layer.weights.cpu().numpy()
        m, n = som_layer.m, som_layer.n
        u_matrix = np.zeros((m, n))
        weights_grid = weights.reshape((m, n, -1))
        
        for i in range(m):
            for j in range(n):
                neighbors = []
                if i > 0: neighbors.append(weights_grid[i-1, j])
                if i < m-1: neighbors.append(weights_grid[i+1, j])
                if j > 0: neighbors.append(weights_grid[i, j-1])
                if j < n-1: neighbors.append(weights_grid[i, j+1])
                
                dists = [np.linalg.norm(weights_grid[i, j] - neigh) for neigh in neighbors]
                u_matrix[i, j] = np.mean(dists)
                
        plt.figure(figsize=(10, 8))
        sns.heatmap(u_matrix, cmap='YlGnBu')
        plt.title(f"{layer_name} - U-Matrix (Cluster Boundaries)")
        plt.savefig(os.path.join(VIS_DIR, f"{layer_name.replace(' ', '_')}_umatrix.png"))
        plt.close()

    @staticmethod
    def plot_hit_map(segment_series, m, n, layer_name):
        hit_map = np.zeros((m, n))
        counts = segment_series.value_counts()
        
        for idx, count in counts.items():
            row = idx // n
            col = idx % n
            if row < m and col < n:
                hit_map[row, col] = count
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(hit_map, cmap='Reds', annot=False)
        plt.title(f"{layer_name} - BMU Hit Map (Density)")
        plt.savefig(os.path.join(VIS_DIR, f"{layer_name.replace(' ', '_')}_hitmap.png"))
        plt.close()

def visualize():
    # 1. Load Mapped Data
    if not os.path.exists(MAPPED_DATA_PATH):
        raise FileNotFoundError("Mapped data not found. Run inference.py first.")
    df_mapped = pd.read_parquet(MAPPED_DATA_PATH)
    print(f"Loaded {len(df_mapped)} mapped records.")

    # 2. Load Model (needed for the weight matrix to generate U-Matrix)
    checkpoint = torch.load(os.path.join(MODEL_DIR, "deep_som.pth"), map_location=DEVICE)
    model = DeepSOM(input_dim=checkpoint['input_dim'], l1_grid=L1_GRID, l2_grid=L2_GRID, max_epochs=EPOCHS)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 3. Generate Visuals
    print("Generating topological maps...")
    vis = SOMVisualizer()
    
    vis.plot_u_matrix(model.layer1, "Layer 1")
    vis.plot_hit_map(df_mapped['segment_l1'], L1_GRID[0], L1_GRID[1], "Layer 1")
    
    vis.plot_u_matrix(model.layer2, "Layer 2")
    vis.plot_hit_map(df_mapped['segment_l2'], L2_GRID[0], L2_GRID[1], "Layer 2")
    
    print(f"Visualizations saved successfully to {os.path.abspath(VIS_DIR)}")

if __name__ == "__main__":
    visualize()