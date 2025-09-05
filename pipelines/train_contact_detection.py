
import time
import pickle
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project's root directory to the Python path
sys.path.append(project_root)

# Now, your imports should work correctly
from src.models.cnn_lstm_contact_detection import cnnLSTM
# ... other imports ...

from src.models.cnn_lstm_contact_detection import cnnLSTM
from src.data_processing import LoadDatasets, LoadSeqDataset
from src.utils import  majority_voting_last_n, contact_detection_accuracy

CONFIG = {
    # General Setup
    'batch_size': 75,
    'robot_name': 'franka_main',
    'source_robot': 'source_franka',
    'dof': 7,
    'split_rate': 0.75,
    'seed': 2020,
    
    # Model Hyperparameters
    'num_layers_list': [1, 2, 3],
    'hidden_size_list': [32, 64, 128, 256],
    'seq_num_list': [30, 50, 80, 100, 150, 200],
    'gap_list': [3, 5, 10, 15],
    'dropout': 0.7,
    
    # Training Parameters
    'learning_rate': 0.004,
    'lr_threshold': 0.0005,
    'n_epochs': 35,
    'loss_fn': nn.BCEWithLogitsLoss(),
    'n_majority_voting': 14,
    
    # Data & Paths
    'main_path': os.getcwd().replace('AIModels', ''),
    'labeled_data_path': 'data/labeled_data', # <-- Added a variable for the labeled data path
    'processed_data_path': 'data/processed_datasets', # <-- Added a variable for the processed data path
    'dict_label': {'link7': 1, 'link6':1, 'link5':1, 'link4':1, 'link3':1, 'link2':1, 'link1':1, 'no_contact': 0},
    'selected_features': [f'e{i}' for i in range(7)],
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name())

# Configure logging
log_file_path = os.path.join(CONFIG['main_path'], f'pipelines/trained_models/{CONFIG["source_robot"]}/contact_detection_{CONFIG["robot_name"]}_training_log_{int(time.time())}.txt')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])

# --- 2. Reusable Functions ---
def train_model(model, train_dataloader, config):
    """Encapsulates the training loop for a given model."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=False)
    
    for epoch in range(config['n_epochs']):
        running_loss = []
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = config['loss_fn'](y_pred, y_batch.float())
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        avg_loss = np.mean(running_loss)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.info(f"Epoch {epoch+1}/{config['n_epochs']} | LR: {current_lr:.5f} | Loss: {avg_loss:.4f}")
        
        if current_lr < config['lr_threshold'] or avg_loss < 0.08:
            logging.info("Early stopping due to low learning rate or loss.")
            break
            
    return model

def evaluate_model(model, config, seq_num=None):
    """Handles the model evaluation and returns performance metrics."""
    model.eval()
    
    # Corrected path using variables from the config
    test_data_path = os.path.join(config['main_path'], config['labeled_data_path'], config['source_robot'])
    test_datasets = LoadDatasets(test_data_path, config['dict_label'])
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)
    
    total_metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    all_contact_delays = []
    all_no_contact_delays = []
    
    for trial_path, label in test_dataloader:
        if 'link1' not in trial_path[0]:
            continue
            
        data = LoadSeqDataset(file_path=trial_path[0], label=label[0], selected_features=config['selected_features'], seq_num=seq_num, gap=1)
        data.sequences = data.sequences[(len(data) // 2):]  # Evaluate on second half
        
        trial_loader = DataLoader(data, batch_size=len(data), shuffle=False)
        
        with torch.no_grad():
            for seqs, labels in trial_loader:
                seqs = seqs.float().to(device)
                predictions = model.prediction(seqs).squeeze().cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                df = pd.DataFrame({'label': labels_np, 'model_out': predictions})
                df['majority_voting'] = np.array([majority_voting_last_n(predictions[:i+1], config['n_majority_voting']) for i in range(len(predictions))])
                
                # Call the helper function to get all metrics
                metrics_df, TP, TN, FP, FN, contact_avg_delay, no_contact_avg_delay, _, _, _, contact_delays, no_contact_delays = contact_detection_accuracy(df)
                
                total_metrics['TP'] += TP
                total_metrics['TN'] += TN
                total_metrics['FP'] += FP
                total_metrics['FN'] += FN
                all_contact_delays.extend([d for d in contact_delays if not np.isnan(d)])
                all_no_contact_delays.extend([d for d in no_contact_delays if not np.isnan(d)])
    
    total_predictions = total_metrics['TP'] + total_metrics['TN'] + total_metrics['FP'] + total_metrics['FN']
    accuracy = (total_metrics['TP'] + total_metrics['TN']) / total_predictions if total_predictions > 0 else 0
    contact_avg_delay = np.mean(all_contact_delays) if all_contact_delays else np.nan
    no_contact_avg_delay = np.mean(all_no_contact_delays) if all_no_contact_delays else np.nan

    return accuracy, contact_avg_delay, no_contact_avg_delay, total_metrics

# --- 3. The Main Execution Loop ---
if __name__ == '__main__':
    # Loop over the source robot's data for training and evaluation
    for seq_num in CONFIG['seq_num_list']:
        for gap in CONFIG['gap_list']:
            # Load data only once per loop!
            try:
                # Corrected processed data path
                processed_data_path = os.path.join(CONFIG["main_path"], CONFIG["processed_data_path"], CONFIG["source_robot"], f'{CONFIG["robot_name"]}_feature_e_gap_{gap}_splitRate_{CONFIG["split_rate"]}_seqNum_{seq_num}.pickle')
                with open(processed_data_path, 'rb') as f:
                    master_dataset = pickle.load(f)
            except FileNotFoundError:
                logging.warning(f"Dataset for {CONFIG['source_robot']} with seq_num={seq_num} and gap={gap} not found. Skipping.")
                continue

            train_dataloader = DataLoader(master_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            
            for num_layers in CONFIG['num_layers_list']:
                for hidden_size in CONFIG['hidden_size_list']:
                    logging.info(f"--- Training: L={num_layers}, H={hidden_size}, S={seq_num}, G={gap} ---")
                    
                    # 1. Instantiate the model
                    model = cnnLSTM(num_features_joints=seq_num, num_layers=num_layers, hidden_size=hidden_size, dropout=CONFIG['dropout'], bidirectional=True).to(device)
                    
                    # 2. Train the model
                    trained_model = train_model(model, train_dataloader, CONFIG)
                    
                    # 3. Evaluate the model
                    accuracy, contact_delay, no_contact_delay, metrics = evaluate_model(trained_model, CONFIG, seq_num)
                    
                    logging.info(f"Accuracy: {accuracy:.2f} | TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
                    logging.info(f"Contact Delay: {contact_delay*1000:.2f}ms | No-Contact Delay: {no_contact_delay*1000:.2f}ms")
                    
                    # 4. Save the trained model with a descriptive filename
                    model_dir = os.path.join(CONFIG['main_path'], f'pipelines/trained_models/{CONFIG["source_robot"]}/contact_detection/{CONFIG["batch_size"]}')
                    os.makedirs(model_dir, exist_ok=True)
                    model_filename = f'L{num_layers}_H{hidden_size}_S{seq_num}_G{gap}_A{accuracy:.2f}.pt'
                    torch.save(trained_model.state_dict(), os.path.join(model_dir, model_filename))
                    logging.info(f"Model saved to {model_dir}/{model_filename}")