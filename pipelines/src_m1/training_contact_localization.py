import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
import os
import sys
import numpy as np
import random
import time
import logging
import glob
from multiprocessing import Pool, cpu_count
import pandas as pd

# --- Add project root to path ---
project_root = os.getcwd().replace('pipelines', '')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipelines.src_m1.dataset_loader import LoadSeqDataset

# --- 1. Model Definition ---
from pipelines.models.cnnLSTM_contactLocalization import cnnLSTM

# --- 2. Helper Functions ---

def load_dataset_worker(args):
    file_path, label, selected_features, seq_num, gap = args
    return LoadSeqDataset(file_path, label, selected_features, seq_num, gap)

def train_localization_model(train_loader, val_loader, model, model_path, n_epochs=20, batch_size=71, learning_rate=0.002):
    """
    Trains the localization model on collision data using CrossEntropyLoss.
    """
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- CHANGE 1: Switched to CrossEntropyLoss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)
    
    best_val_accuracy = 0.0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            
            # --- CHANGE 2: Prepare labels for CrossEntropyLoss (0-indexed, long type) ---
            # No longer need create_hierarchical_labels function
            target_labels = (labels - 1).long()
            loss = criterion(outputs, target_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                
                # --- Also use the correct label format for validation loss ---
                target_labels = (labels - 1).long()
                val_loss += criterion(outputs, target_labels).item()
                
                # --- CHANGE 3: Update accuracy calculation for 0-indexed labels ---
                predicted = torch.argmax(outputs, dim=1)
                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        logging.info(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        
        scheduler.step(avg_val_loss)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            logging.info(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 0.0005:
            logging.info("LR dropped below threshold. Stopping early.")
            break
        
        

        
    return best_val_accuracy

if __name__ == '__main__':
    # --- Main Configuration ---
    data_name = 'franka_mindlab'
    dof = 7

    hidden_sizes = [32, 64, 128, 256, 512, 1024]
    num_layers_list = [1, 2, 3]
    seq_nums = [30, 50, 80, 100, 150, 200, 250, 300]
    gaps = [3, 5, 10]

    batch_size = 64
    n_epochs = 40

    log_dir = f'{project_root}/pipelines/trained_models/{data_name}/m1_cnnBiLSTM_contactLocalization/{batch_size}/'

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_localization_{time.time()}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    logging.info(f"Logging to {log_file}")

    results_list = []
    results_csv_path = os.path.join(log_dir, 'hyperparameter_results_localization.csv')

    torch.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)

    
    # --- Hyperparameter Training Loop ---
    for gap in gaps:
        for seq_num in seq_nums:
            logging.info(f"Starting data loading for seq_num={seq_num}, gap={gap}")
            
            # --- Load Data in Parallel for this configuration ---
            data_directory = os.path.join(project_root, 'dataset', data_name, 'labeled_data')
            dict_label = {'link7': 7, 'link6': 6, 'link5': 5, 'link4': 4, 'link3': 3, 'link2': 2, 'link1': 1, 'no_contact': 0}
            selected_features = [f'e{i}' for i in range(dof)]
            
            all_csv_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)
    
            tasks = []
            for file in all_csv_files:
                label_val = 0
                for name, number in dict_label.items():
                    if name in file:
                        label_val = number
                        break
                tasks.append((file, torch.tensor(label_val), selected_features, seq_num, gap))

            with Pool(processes=cpu_count()) as pool:
                all_datasets = pool.map(load_dataset_worker, tasks)
            
            master_dataset = ConcatDataset(all_datasets)

            collision_indices = [i for i, (_, label) in enumerate(master_dataset) if label > 0]
            if not collision_indices:
                logging.warning(f"No collision data for seq_num={seq_num}. Skipping.")
                break
            
            collision_dataset = Subset(master_dataset, collision_indices)

            train_size = int(0.5 * len(collision_dataset))
            val_size = len(collision_dataset) - train_size
            train_dataset, val_dataset = random_split(collision_dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)

            for hidden_size in hidden_sizes:
                for num_layers in num_layers_list:
                    model = cnnLSTM(num_features_joints=seq_num, hidden_size=hidden_size, num_layers=num_layers)
                    model_name = f'numLayer{num_layers}_hiddenSize{hidden_size}_seq_num{seq_num}_gap{gap}'
                
                    logging.info(f"--- Training Localization: hidden={hidden_size}, layers={num_layers}, seq={seq_num} ---")
                    best_accuracy = train_localization_model(train_loader=train_loader, val_loader=val_loader, model=model, 
                                                             model_path = f'{log_dir}{model_name}.pth', batch_size=batch_size, n_epochs=n_epochs, learning_rate=0.001)
                    
                    os.rename(f'{log_dir}{model_name}.pth', f'{log_dir}{model_name}_accuracy{best_accuracy:.2f}.pth')
                    results_list.append({
                        'gap': gap, 'seq_num': seq_num, 'hidden_size': hidden_size,
                        'num_layer': num_layers, 'ACC': best_accuracy, 
                        'train_size_localization': train_size, 'train_size_detection': len(master_dataset)
                    })
                    results_df = pd.DataFrame(results_list)
                    results_df.to_csv(results_csv_path, index=False)
                    logging.info(f"Results updated at {results_csv_path}")

    logging.info("\n--- Hyperparameter search for localization complete ---")