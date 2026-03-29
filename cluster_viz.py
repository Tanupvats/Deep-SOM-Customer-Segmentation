import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Configuration
MAPPED_DATA_PATH = "./models/mapped_data.parquet"
VIS_DIR = "./visualizations"
SAMPLE_SIZE = 10_000  # Dimensionality reduction is computationally heavy; we sample the data

os.makedirs(VIS_DIR, exist_ok=True)

def load_and_sample_data():
    """Loads the mapped data and samples it for visualization."""
    print("Loading mapped data...")
    if not os.path.exists(MAPPED_DATA_PATH):
        raise FileNotFoundError(f"Cannot find {MAPPED_DATA_PATH}. Run inference first.")
    
    df = pd.read_parquet(MAPPED_DATA_PATH)
    
    print(f"Sampling {SAMPLE_SIZE} records from {len(df)} total records...")
    df_sampled = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).copy()
    return df_sampled

def prepare_features(df_sampled):
    """Selects numeric features and scales them for PCA/t-SNE."""
    numeric_features = [
        'age', 'estimated_annual_income', 'credit_score', 
        'mobile_app_logins_30d', 'push_notification_ctr', 
        'active_products_count', 'total_deposit_balance', 'loan_inquiries_12m'
    ]
    
    # Impute NaNs for visualization purposes
    X = df_sampled[numeric_features].fillna(df_sampled[numeric_features].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def visualize_2d_pca(X_scaled, labels, layer_name):
    """Projects data to 2D using PCA and saves a scatter plot."""
    print(f"Generating 2D PCA for {layer_name}...")
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=components[:, 0], y=components[:, 1],
        hue=labels, palette="tab20", s=30, alpha=0.7, legend="full"
    )
    plt.title(f"2D PCA Projection of {layer_name} Clusters")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    
    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"2d_pca_{layer_name.replace(' ', '_')}.png"))
    plt.close()

def visualize_3d_pca_interactive(df_sampled, X_scaled, layer_name, segment_col):
    """Projects data to 3D using PCA and generates an interactive HTML plot."""
    print(f"Generating interactive 3D PCA for {layer_name}...")
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    
    # Add components to the dataframe for Plotly
    df_plot = df_sampled.copy()
    df_plot['PC1'] = components[:, 0]
    df_plot['PC2'] = components[:, 1]
    df_plot['PC3'] = components[:, 2]
    
    # Ensure segment column is treated as categorical/string for discrete colors
    df_plot[segment_col] = df_plot[segment_col].astype(str)
    
    fig = px.scatter_3d(
        df_plot, x='PC1', y='PC2', z='PC3',
        color=segment_col,
        hover_data=['user_type', 'age', 'estimated_annual_income', 'credit_score'],
        title=f"Interactive 3D PCA Projection of {layer_name} Clusters",
        opacity=0.7,
        size_max=5
    )
    
    # Make points slightly smaller for better visibility
    fig.update_traces(marker=dict(size=3))
    
    # Save as interactive HTML
    html_path = os.path.join(VIS_DIR, f"3d_interactive_{layer_name.replace(' ', '_')}.html")
    fig.write_html(html_path)
    print(f"  -> Saved interactive 3D plot to {html_path}")

def main():
    df_sampled = load_and_sample_data()
    X_scaled = prepare_features(df_sampled)
    
    # Group into exactly 10 macro-clusters using K-Means
    print("Grouping data into exactly 10 macro-clusters for visualization...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    df_sampled['macro_cluster_10'] = kmeans.fit_predict(X_scaled)
    
    # Visualize the 10 Macro Clusters
    visualize_2d_pca(X_scaled, df_sampled['macro_cluster_10'], "10 Macro Clusters")
    visualize_3d_pca_interactive(df_sampled, X_scaled, "10 Macro Clusters", "macro_cluster_10")
    
    print(f"\nAll visualizations have been generated and saved to the '{VIS_DIR}' directory.")
    print("Open the generated .html files in your web browser to explore the 3D plots!")

if __name__ == "__main__":
    main()