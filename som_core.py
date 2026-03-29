import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DATA_DIR = "./data"
MODEL_DIR = "./models"
VIS_DIR = "./visualizations"
MAPPED_DATA_PATH = os.path.join(MODEL_DIR, "mapped_data.parquet")

BATCH_SIZE = 8192
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SOM Hyperparameters
L1_GRID = (20, 20)
L2_GRID = (10, 10)
EPOCHS = 30

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
class DataPipeline:
    def __init__(self):
        self.numeric_features = [
            'age', 'estimated_annual_income', 'credit_score', 
            'mobile_app_logins_30d', 'push_notification_ctr', 
            'active_products_count', 'total_deposit_balance', 'loan_inquiries_12m'
        ]
        self.categorical_features = ['region', 'employment_status', 'user_type']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        self.is_fit = False

    def fit_transform(self, df):
        processed = self.preprocessor.fit_transform(df)
        self.is_fit = True
        return processed

    def transform(self, df):
        if not self.is_fit:
            raise RuntimeError("Pipeline must be fitted before calling transform().")
        return self.preprocessor.transform(df)

# ==========================================
# 3. PYTORCH MINI-BATCH SOM
# ==========================================
class MiniBatchSOM(nn.Module):
    def __init__(self, m, n, dim, max_epochs, init_lr=0.5):
        super().__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.max_epochs = max_epochs
        self.init_lr = init_lr
        self.init_sigma = max(m, n) / 2.0
        
        self.weights = nn.Parameter(torch.randn(m * n, dim), requires_grad=False)
        self.locations = self._get_grid_locations().to(DEVICE)

    def _get_grid_locations(self):
        x, y = torch.meshgrid(torch.arange(self.m), torch.arange(self.n), indexing='ij')
        return torch.stack([x.flatten(), y.flatten()], dim=1).float()

    def forward(self, x, current_epoch):
        dist = torch.cdist(x, self.weights)
        bmu_indices = torch.argmin(dist, dim=1)
        bmu_locs = self.locations[bmu_indices]
        
        if self.training:
            decay_factor = torch.exp(torch.tensor(-current_epoch / self.max_epochs))
            lr = self.init_lr * decay_factor
            sigma = self.init_sigma * decay_factor
            
            bmu_node_dist = torch.cdist(bmu_locs, self.locations)
            h = torch.exp(-(bmu_node_dist**2) / (2 * sigma**2))
            
            num = torch.matmul(h.t(), x)
            den = h.sum(dim=0).unsqueeze(1)
            den[den == 0] = 1e-8
            
            new_weights = num / den
            self.weights.data = self.weights.data * (1 - lr) + new_weights * lr
            
        return bmu_indices, self.weights[bmu_indices]

class DeepSOM(nn.Module):
    def __init__(self, input_dim, l1_grid, l2_grid, max_epochs):
        super().__init__()
        self.layer1 = MiniBatchSOM(l1_grid[0], l1_grid[1], input_dim, max_epochs)
        self.layer2 = MiniBatchSOM(l2_grid[0], l2_grid[1], input_dim, max_epochs)

    def forward(self, x, epoch):
        l1_bmu_idx, l1_bmu_weights = self.layer1(x, epoch)
        l2_bmu_idx, _ = self.layer2(l1_bmu_weights, epoch)
        return l1_bmu_idx, l2_bmu_idx