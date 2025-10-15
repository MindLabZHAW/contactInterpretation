import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
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

from pipelines.src_m2.dataset_loader import LoadSeqDataset
from pipelines.models.cnnLSTM_contactLocalization import cnnLSTM

# --- Helper Functions (Unchanged) ---
def load_dataset_worker(args):
    file_path, label_val, selected_features, mode, seq_num, gap = args
    # Note: The LoadSeqDataset now expects a 'mode' argument
    return LoadSeqDataset(file_path, label_val, selected_features, mode, seq_num, gap)

def train_localization_model(train_loader, val_loader, model, model_path, n_epochs=20, learning_rate=0.002):
    """
    Trains the localization model on collision data using CrossEntropyLoss.
    (This function is unchanged)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)
    best_val_accuracy = 0.0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            target_labels = (labels - 1).long() # Convert labels 1-7 to 0-6
            loss = criterion(outputs, target_labels)
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
                target_labels = (labels - 1).long()
                val_loss += criterion(outputs, target_labels).item()
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
        if current_lr < 1e-5: # Adjusted threshold
            logging.info("LR dropped below threshold. Stopping early.")
            break
    return best_val_accuracy

if __name__ == '__main__':
    # --- Main Configuration ---
    data_name = 'ur5'
    dof = 6
    hidden_sizes = [32, 64, 128, 256]#, 512, 1024]
    num_layers_list = [1, 2, 3]
    seq_nums = [30, 50, 80, 100]#, 150, 200, 250, 300, 350, 400, 450, 500]
    #seq_nums = [450, 500]

    gaps = [1]#[3, 5, 10]
    batch_size = 65
    n_epochs = 40
    
    # --- Define a fixed gap for the validation set ---
    VALIDATION_GAP = 5

    log_dir = f'{project_root}/pipelines/trained_models/{data_name}/m2_cnnBiLSTM_contactLocalization/{batch_size}/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_localization_{time.time()}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    logging.info(f"Logging to {log_file}")

    results_list = []
    results_csv_path = os.path.join(log_dir, 'hyperparameter_results_localization.csv')
    
    # --- Data Loading Setup ---
    data_directory = os.path.join(project_root, 'dataset', data_name, 'labeled_data')
    dict_label = {'link7': 7, 'link6': 6, 'link5': 5, 'link4': 4, 'link3': 3, 'link2': 2, 'link1': 1}
    selected_features = [f'e{i}' for i in range(dof)]
    all_csv_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)

    if not all_csv_files:
        logging.error(f"No CSV files found in '{data_directory}'.")
    else:
        # --- RESTRUCTURED LOOP LOGIC ---
        for seq_num in seq_nums:
            logging.info(f"--- Processing for Sequence Length (seq_num) = {seq_num} ---")
            
            with Pool(processes=cpu_count()) as pool:
                # 1. Load the STABLE VALIDATION dataset (first half of files)
                val_tasks = []
                for file in all_csv_files:
                    label_val = next((num for name, num in dict_label.items() if name in file), 0)
                    if label_val > 0: # Only create tasks for files with contact
                        val_tasks.append((file, torch.tensor(label_val), selected_features, 'val', seq_num, VALIDATION_GAP))
                
                logging.info(f"Loading validation datasets with fixed gap={VALIDATION_GAP}...")
                all_val_datasets = pool.map(load_dataset_worker, val_tasks)
                val_master_dataset = ConcatDataset(all_val_datasets)

                # Filter for collision data only
                val_collision_indices = [i for i, (_, label) in enumerate(val_master_dataset) if label > 0]
                val_dataset = Subset(val_master_dataset, val_collision_indices)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)
                logging.info(f"Created stable Validation Dataset for localization with {len(val_dataset)} samples.")

                # 2. Loop through training gaps to create different TRAINING datasets
                for train_gap in gaps:
                    logging.info(f"--- Processing for Training Gap = {train_gap} ---")
                    
                    train_tasks = []
                    for file in all_csv_files:
                        label_val = next((num for name, num in dict_label.items() if name in file), 0)
                        if label_val > 0: # Only create tasks for files with contact
                            train_tasks.append((file, torch.tensor(label_val), selected_features, 'train', seq_num, train_gap))

                    logging.info(f"Loading training datasets with gap={train_gap}...")
                    all_train_datasets = pool.map(load_dataset_worker, train_tasks)
                    train_master_dataset = ConcatDataset(all_train_datasets)

                    # Filter for collision data only
                    train_collision_indices = [i for i, (_, label) in enumerate(train_master_dataset) if label > 0]
                    train_dataset = Subset(train_master_dataset, train_collision_indices)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True)
                    logging.info(f"Created Training Dataset for localization with {len(train_dataset)} samples.")

                    # 3. Finally, loop through model architectures and train
                    for hidden_size in hidden_sizes:
                        for num_layers in num_layers_list:
                            # NOTE: The first argument should be the number of features (dof), not the sequence length.
                            model = cnnLSTM(num_features_joints=seq_num, hidden_size=hidden_size, num_layers=num_layers)
                            model_name = f'numLayer{num_layers}_hiddenSize{hidden_size}_seq_num{seq_num}_gap{train_gap}'
                            
                            logging.info(f"--- Training Localization: hidden={hidden_size}, layers={num_layers}, seq={seq_num}, gap={train_gap} ---")
                            best_accuracy = train_localization_model(
                                train_loader=train_loader, val_loader=val_loader, model=model,
                                model_path=f'{log_dir}{model_name}.pth', n_epochs=n_epochs, learning_rate=0.001
                            )
                            
                            if os.path.exists(f'{log_dir}{model_name}.pth'):
                                os.rename(f'{log_dir}{model_name}.pth', f'{log_dir}{model_name}_accuracy{best_accuracy:.2f}.pth')
                            
                            results_list.append({
                                'seq_num': seq_num, 'gap': train_gap, 'hidden_size': hidden_size,
                                'num_layer': num_layers, 'ACC': best_accuracy,
                                'train_size': len(train_dataset), 'val_size': len(val_dataset)
                            })
                            results_df = pd.DataFrame(results_list)
                            results_df.to_csv(results_csv_path, index=False)
                            logging.info(f"Results updated at {results_csv_path}")

        logging.info("\n--- Hyperparameter search for localization complete ---")