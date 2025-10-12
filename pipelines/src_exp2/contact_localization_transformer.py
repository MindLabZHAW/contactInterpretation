import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import os
import sys
import math
import time
import logging
import glob
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm


# --- Add project root to path ---
project_root = os.getcwd().replace('pipelines', '')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from pipelines.src_exp2.dataset_loader import LoadSeqDataset
    from pipelines.models.transformer_contactLocalization import PositionalEncoding, TransformerForLinkLocalization
except ImportError:
    print("Please ensure your project structure and paths are set up correctly.")
    sys.exit(1)

# --- Helper Functions (Unchanged) ---
def load_dataset_worker(args):
    file_path, label_val, selected_features, mode, seq_num, gap = args
    return LoadSeqDataset(file_path, label_val, selected_features, mode, seq_num, gap)

def train_localization_model(train_loader, val_loader, model, model_path, n_epochs=20, learning_rate=0.0001):
    """
    Trains the localization model on collision data using CrossEntropyLoss.
    (This function is unchanged and compatible with the new model)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_val_accuracy = 0.0
    logging.info("--- Starting Optimized Model Training ---")

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [T]", unit="batch")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
            target_labels = (labels - 1).long()
            loss = criterion(outputs, target_labels)

            if torch.isnan(loss):
                #logging.warning(f"NaN loss detected at epoch {epoch+1}. Skipping batch.")
                continue
            
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            running_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [V]", unit="batch")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                
                target_labels = (labels - 1).long()
                val_loss += criterion(outputs, target_labels).item()
                _, predicted = torch.max(outputs.data, 1)
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
        if current_lr < 25e-6:
            logging.info("LR dropped below threshold. Stopping early.")
            break
    return best_val_accuracy

if __name__ == '__main__':
    # --- Main Configuration ---
    data_name = 'ur5'
    dof = 6
    batch_size = 64
    n_epochs = 40

    seq_nums = [ 100]    
    gaps = [1]
    d_models = [64, 128, 256, 512]  # Test a larger model
    n_heads = [1, 4, 8]       # Test more attention heads
    num_encoder_layers_list = [1,4, 8]
    dropout_rates = [0.3]
    lr_rate = 0.0001
    
    VALIDATION_GAP = 5

    log_dir = f'{project_root}/pipelines/trained_models/{data_name}/contact_localization_transformer/{batch_size}/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_localization_transformer_{time.time()}.txt')
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
        for seq_num in seq_nums:
            logging.info(f"--- Processing for Sequence Length (num_features) = {seq_num} ---")
            
            with Pool(processes=cpu_count()) as pool:
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



                    # --- ✅ 3. UPDATED INNER LOOP FOR TRANSFORMER HYPERPARAMETERS ---
                    for d_model in d_models:
                        for n_head in n_heads:
                            if d_model % n_head != 0:
                                continue
                            for num_layers in num_encoder_layers_list:
                                for dropout in dropout_rates:
                                    model = TransformerForLinkLocalization(
                                        num_features=seq_num,
                                        d_model=d_model,
                                        nhead=n_head,
                                        num_encoder_layers=num_layers,
                                        dim_feedforward=d_model * 4,
                                        dropout=dropout                                    )
                                    
                                    model_name = f'dModel{d_model}_nHead{n_head}_nLayer{num_layers}_sNum{seq_num}'

                                    logging.info(f"--- Training: {model_name} ---")
                                    
                                    best_accuracy = train_localization_model(
                                        train_loader=train_loader, val_loader=val_loader, model=model,
                                        model_path=f'{log_dir}{model_name}.pth', n_epochs=n_epochs, learning_rate=lr_rate
                                    )
                                    
                                    final_model_path = f'{log_dir}{model_name}_acc{best_accuracy:.2f}.pth'
                                    if os.path.exists(f'{log_dir}{model_name}.pth'):
                                        os.rename(f'{log_dir}{model_name}.pth', final_model_path)
                                    
                                    results_list.append({
                                        'seq_num': seq_num, 'gap': train_gap, 'd_model': d_model, 'n_head': n_head,
                                        'num_layer': num_layers, 'dropout': dropout, 'ACC': best_accuracy,
                                        'train_size': len(train_master_dataset), 'val_size': len(val_master_dataset)
                                    })
                                    pd.DataFrame(results_list).to_csv(results_csv_path, index=False)
                                    logging.info(f"Results updated at {results_csv_path}")

        logging.info("\n--- Hyperparameter search for localization complete ---")