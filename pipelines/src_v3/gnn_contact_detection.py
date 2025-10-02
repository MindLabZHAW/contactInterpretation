# gnn_robot_contact.py
#
# This script builds and trains a Graph Neural Network (GNN) for binary
# contact detection. It uses a custom torch.utils.data.Dataset to load
# and process the data, with a time window of features for each node.
#
# --- Data Structure Expectation ---
# The script expects data in a directory like 'dataset/franka_main/labeled_data/', organized as:
#
#   dataset/franka_main/labeled_data/
#   ├── no_contact/
#   │   ├── sample1.csv
#   │   └── ...
#   ├── link1/
#   │   ├── sample1.csv
#   │   └── ...
#   └── ...
#
# CRITICAL: Each data file MUST contain a 'label' column and feature columns
#           (e.g., e0..e6). This script will convert labels > 0 to 1 for
#           binary classification.
#
# Installation:
# pip install torch torch-geometric pandas scikit-learn

import os
import glob
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import BatchNorm1d, Linear
from sklearn.utils.class_weight import compute_class_weight

# --- 1. Custom Robot Graph Dataset Class (using torch.utils.data.Dataset) ---
class RobotGraphTimeDataset(Dataset):
    """
    A custom PyTorch Dataset that processes time-series files into graph samples.
    It splits the CONTENT of each file into train/val/test sets.
    """
    def __init__(self, root_dir, mode='train', window_size=20, train_split=0.7, val_split=0.15):
        self.window_size = window_size
        self.mode = mode
        self.train_split = train_split
        self.val_split = val_split
        
        # --- Define static graph structure ---
        dof = 7
        edge_list = [[i, i + 1] for i in range(dof - 1)] + [[i + 1, i] for i in range(dof - 1)]
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.feature_cols = [f'e{i}' for i in range(dof)]
        
        # --- Find all data files ---
        self.file_paths = self._find_files(root_dir)
        print(f"[{self.mode.upper()} SET] Found {len(self.file_paths)} files to process.")
        
        # --- Pre-process data for this split into memory ---
        self.graph_samples = self._create_graph_windows()

    def _find_files(self, root_dir):
        """Finds all relevant .csv and .pkl files in the data directory."""
        labeled_data_dir = os.path.join(root_dir, 'labeled_data')
        all_file_paths = glob.glob(os.path.join(labeled_data_dir, '**', '*.csv'), recursive=True)
        all_file_paths.extend(glob.glob(os.path.join(labeled_data_dir, '**', '*.pkl'), recursive=True))
        return sorted(list(set(all_file_paths)))

    def _create_graph_windows(self):
        """Reads files and creates a list of graph window samples from the correct data split."""
        samples = []
        for file_path in self.file_paths:
            try:
                df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_pickle(file_path)
                if 'label' not in df.columns or not all(f in df.columns for f in self.feature_cols):
                    continue
            except Exception:
                continue
            
            # --- NEW: Split the content of the dataframe ---
            num_rows = len(df)
            train_end = int(self.train_split * num_rows)
            val_end = int((self.train_split + self.val_split) * num_rows)

            if self.mode == 'train':
                df_split = df.iloc[:train_end]
            elif self.mode == 'val':
                df_split = df.iloc[train_end:val_end]
            else: # test
                df_split = df.iloc[val_end:]

            # --- Create windows from the data split ---
            for i in range(self.window_size - 1, len(df_split)):
                window_df = df_split.iloc[i - self.window_size + 1 : i + 1]
                
                features = window_df[self.feature_cols].values.T
                label = int(window_df['label'].iloc[-1])
                binary_label = 1 if label > 0 else 0
                
                samples.append((features, binary_label))
        
        print(f"[{self.mode.upper()} SET] Created {len(samples)} graph samples.")
        return samples

    def __len__(self):
        return len(self.graph_samples)

    def __getitem__(self, idx):
        features_np, label = self.graph_samples[idx]
        
        node_features = torch.tensor(features_np, dtype=torch.float)
        graph_label = torch.tensor([label], dtype=torch.long)
        
        return Data(x=node_features, edge_index=self.edge_index, y=graph_label)

# --- 2. Upgraded GNN Model ---
class GATNet(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, num_classes):
        super(GATNet, self).__init__()
        torch.manual_seed(42)
        
        # --- FIX: Instantiated GATConv, not GATNet ---
        self.conv1 = GATConv(input_features, hidden_channels, heads=4)
        self.bn1 = BatchNorm1d(hidden_channels * 4)
        
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=2)
        self.bn2 = BatchNorm1d(hidden_channels * 4)

        self.lin1 = Linear(hidden_channels * 4, hidden_channels * 2)
        self.lin2 = Linear(hidden_channels * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x

# --- 3. Main Training & Evaluation Block ---
if __name__ == '__main__':
    # --- A. Setup Your Data Directory & Hyperparameters ---
    data_name = 'franka_mindlab'
    data_dir = f'dataset/{data_name}/'
    WINDOW_SIZE = 200
    hidden_size = 128
    
    # --- B. Create Datasets and DataLoaders ---
    print("\n--- Step B: Loading Robot Graph Dataset ---")
    # This will create and process each dataset split individually
    train_dataset = RobotGraphTimeDataset(root_dir=data_dir, mode='train', window_size=WINDOW_SIZE)
    val_dataset = RobotGraphTimeDataset(root_dir=data_dir, mode='val', window_size=WINDOW_SIZE)
    test_dataset = RobotGraphTimeDataset(root_dir=data_dir, mode='test', window_size=WINDOW_SIZE)
    
    batch_size = 256
    # Use torch_geometric.loader.DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # --- C. Initialize and Train the Model ---
    print("\n--- Step C: Training the GNN ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The number of input features is now the window size
    model = GATNet(
        input_features=WINDOW_SIZE,
        hidden_channels=hidden_size,
        num_classes=2
    ).to(device)
    print(f"Model initialized with {WINDOW_SIZE} input features per node.")
    
    # Calculate class weights from the training set only
    train_labels = [label for _, label in train_dataset.graph_samples]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Calculated class weights for binary problem:\n{class_weights}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)

    best_val_acc = 0.0
    save_dir = f'pipelines/trained_models/{data_name}/contact_detection/gnn/'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'hiddenSize{hidden_size}_ws{WINDOW_SIZE}.pth')

    def train_loop():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(train_loader.dataset)

    def test_loop(loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == batch.y).sum())
        return correct / len(loader.dataset) * 100

    for epoch in range(1, 101):
        loss = train_loop()
        val_acc = test_loop(val_loader)
        scheduler.step(val_acc)
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'*** New best model saved with Val Accuracy: {val_acc:.4f} at Epoch {epoch:03d} ***')
    os.rename(model_path, model_path.replace('.pth',f'_accutacy{best_val_acc:.2f}.pth'))
    model_path= model_path.replace('.pth',f'_accutacy{best_val_acc}.pth')
    # --- D. Final Evaluation ---
    print("\n--- Step D: Final Evaluation ---")
    model.load_state_dict(torch.load(model_path))
    model.val()
    test_acc = test_loop(test_loader)
    print(f'Final Test Accuracy (using best model): {test_acc:.4f}')

