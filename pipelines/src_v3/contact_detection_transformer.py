import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os
import logging
import glob
from multiprocessing import Pool, cpu_count
import time
import sys
import pandas as pd

# --- Add project root to path ---
project_root = os.getcwd().replace('pipelines', '')

from src_v3.dataset_loader import LoadSeqDataset
from models.transformer_contactDetection import TransformerModel

# Helper function for parallel data loading (unchanged)
def load_dataset_worker(args):
    """Worker function for parallel data loading."""
    file_path, label, selected_features, mode, seq_num, gap = args
    return LoadSeqDataset(file_path, label, selected_features, mode, seq_num, gap)

# train_model function
def train_model(train_loader, val_loader,model,  n_epochs=50, batch_size=64, learning_rate=0.0001, model_path='transformer_model'):
    """
    Trains and validates the model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val_accuracy = 0.0

    logging.info("--- Starting Optimized Model Training ---")
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Check for nan loss before backpropagation
            if torch.isnan(loss):
                logging.warning(f"NaN loss detected at epoch {epoch+1}. Skipping batch.")
                continue

            optimizer.zero_grad(set_to_none=True)

            scaler_amp.scale(loss).backward()
            # --- THE FIX: GRADIENT CLIPPING ---
            # Unscale the gradients before clipping to avoid issues with mixed precision
            scaler_amp.unscale_(optimizer)
            # Clip the gradients to a maximum norm of 1.0. This prevents them from exploding.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
                
                # Permute input tensor dimensions (already present here, which is correct)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                
                val_loss += criterion(outputs, labels).item()
                # The prediction needs the same shaped input
                predicted = (torch.sigmoid(outputs) > 0.5).int()
                total += labels.size(0)
                # Ensure predicted and labels are on the same device for comparison
                correct += (predicted.cpu() == labels.cpu().int()).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        logging.info(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        scheduler.step(avg_val_loss)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{model_path}')
            logging.info(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 0.00005:
            logging.info(f"Learning rate ({current_lr:.6f}) has dropped below the threshold. Stopping training early.")
            break
            
    return model, best_val_accuracy


if __name__ == '__main__':
    # --- Configuration ---
    project_root = os.getcwd().replace('pipelines','')
    data_name = 'franka_main'
    dof = 7
    batch_size = 64
    n_epochs = 40
    VALIDATION_GAP = 1
    
    # --- Hyperparameters for the Transformer model ---
    d_models = [128]
    n_heads = [4]
    num_encoder_layers_list = [2]
    seq_nums = [300]
    gaps = [1]

    log_dir = f'{project_root}/pipelines/trained_models/{data_name}/contact_detection_transformer/{batch_size}/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    logging.info(f"Logging to {log_file}")
    
    # --- Data Loading Setup ---
    data_directory = os.path.join(project_root, 'dataset', data_name, 'labeled_data')
    selected_features = [f'e{i}' for i in range(dof)]
    all_csv_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)
    results_list = []
    results_csv_path = os.path.join(log_dir, 'hyperparameter_results.csv')

    if not all_csv_files:
        logging.error(f"No CSV files found in '{data_directory}'. Please check the path.")
    else:
        for seq_num in seq_nums:
            logging.info(f"--- Processing for Sequence Length (seq_num) = {seq_num} ---")

            with Pool(processes=cpu_count()) as pool:
                val_tasks = [(file, torch.tensor(1 if 'no_contact' not in file else 0), selected_features, 'val', seq_num, VALIDATION_GAP) for file in all_csv_files]
                
                logging.info(f"Loading validation datasets with fixed gap={VALIDATION_GAP}...")
                all_val_datasets = pool.map(load_dataset_worker, val_tasks)
                
                val_dataset = ConcatDataset(all_val_datasets)
                val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=min(4, os.cpu_count()), pin_memory=True, persistent_workers=True)
                logging.info(f"Created stable Validation Dataset for seq_num={seq_num} with {len(val_dataset)} samples.")

                for train_gap in gaps:
                    logging.info(f"--- Processing for Training Gap = {train_gap} ---")
                    
                    train_tasks = [(file, torch.tensor(1 if 'no_contact' not in file else 0), selected_features, 'train', seq_num, train_gap) for file in all_csv_files]

                    logging.info(f"Loading training datasets with gap={train_gap}...")
                    all_train_datasets = pool.map(load_dataset_worker, train_tasks)

                    train_dataset = ConcatDataset(all_train_datasets)
                    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=min(4, os.cpu_count()), pin_memory=True, persistent_workers=True)
                    logging.info(f"Created Training Dataset for gap={train_gap} with {len(train_dataset)} samples.")

                    for d_model in d_models:
                        for n_head in n_heads:
                            # Ensure d_model is divisible by n_head
                            if d_model % n_head != 0:
                                logging.warning(f"Skipping combination: d_model={d_model} is not divisible by n_head={n_head}")
                                continue
                                
                            for num_layers in num_encoder_layers_list:
                                logging.info(f"Starting training for: seq_num={seq_num}, gap={train_gap}, d_model={d_model}, n_head={n_head}, num_layers={num_layers}")
                                
                                # --- FIX #2: Initialize the TransformerModel with the correct number of features ---
                                model = TransformerModel(num_features=seq_num,
                                                         d_model=d_model, 
                                                         nhead=n_head, 
                                                         num_encoder_layers=num_layers,
                                                         dim_feedforward=d_model*4,
                                                         dropout=0.5)
                                
                                model_name = f'd_model{d_model}_n_head{n_head}_numLayer{num_layers}_seq_num{seq_num}_gap{train_gap}'

                                trained_model, accuracy = train_model(train_loader=train_loader, val_loader=val_loader,
                                                                      model=model, n_epochs=n_epochs,
                                                                      model_path=f'{log_dir}{model_name}.pth')
                                
                                final_model_path = f'{log_dir}{model_name}_accuracy{accuracy:.2f}.pth'
                                if os.path.exists(f'{log_dir}{model_name}.pth'):
                                    os.rename(f'{log_dir}{model_name}.pth', final_model_path)
                                
                                results_list.append({
                                    'gap': train_gap,
                                    'seq_num': seq_num,
                                    'd_model': d_model,
                                    'n_head': n_head,
                                    'num_layer': num_layers,
                                    'ACC': accuracy,
                                    'train_size': len(train_dataset),
                                    'val_size': len(val_dataset)
                                })
                                results_df = pd.DataFrame(results_list)
                                results_df.to_csv(results_csv_path, index=False)
                                logging.info(f"Results updated at {results_csv_path}")
                                
    logging.info("\n--- Hyperparameter search complete ---")