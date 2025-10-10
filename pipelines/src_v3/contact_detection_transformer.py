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
from tqdm import tqdm

# --- Add project root to path ---
project_root = os.getcwd().replace('pipelines', '')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from pipelines.src_v3.dataset_loader import LoadSeqDataset
    from pipelines.models.transformer_contactDetection import TransformerModel # Assuming the model is in transformer_model.py
except ImportError:
    print("Please ensure your project structure and paths are set up correctly.")
    sys.exit(1)



# --- Worker function for parallel data loading (unchanged) ---
def load_dataset_worker(args):
    """Worker function for parallel data loading."""
    file_path, label, selected_features, mode, seq_num, gap = args
    return LoadSeqDataset(file_path, torch.tensor(label), selected_features, mode, seq_num, gap)

# --- train_model function with added weight_decay ---
def train_model(train_loader, val_loader, model, n_epochs=50, learning_rate=0.0001, weight_decay=1e-4, model_path='transformer_model'):
    """
    Trains and validates the model, now with weight decay.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    # Use AdamW optimizer which is standard for Transformers
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
    
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val_accuracy = 0.0

    logging.info("--- Starting Optimized Model Training ---")
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [T]", unit="batch")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                #logging.warning(f"NaN loss detected at epoch {epoch+1}. Skipping batch.")
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            running_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [V]", unit="batch")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                
                val_loss += criterion(outputs, labels).item()
                predicted = (torch.sigmoid(outputs) > 0.5).int()
                total += labels.size(0)
                correct += (predicted == labels.int()).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        logging.info(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        scheduler.step(avg_val_loss)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{model_path}')
            logging.info(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-6:
            logging.info(f"Learning rate ({current_lr:.6f}) is very low. Stopping training early.")
            break
            
    return model, best_val_accuracy


if __name__ == '__main__':
    # --- Configuration ---
    project_root = os.getcwd().replace('pipelines','')
    data_name = 'franka_main'
    dof = 7
    batch_size = 67
    n_epochs = 30
    VALIDATION_GAP = 5
    
    # --- EXPANDED Hyperparameter Search Space ---
    d_models = [1024]#[64, 128, 256, 512, 1024]  # Test a larger model
    n_heads = [2, 4, 8]       # Test more attention heads
    num_encoder_layers_list = [1, 4, 8] # Test a deeper model
    dropout_rates = [0.3] # Tune dropout for regularization
    learning_rates = [0.0001] # Can expand this too, e.g., [0.0001, 0.00005]
    weight_decays = [1e-4] # Tune weight decay
    
    seq_nums = [200]#[100, 200, 300]
    gaps = [1]

    log_dir = f'{project_root}/pipelines/trained_models/{data_name}/contact_detection_transformer/{batch_size}/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_hyperparam_search_{time.strftime("%Y%m%d-%H%M%S")}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logging.info(f"Logging to {log_file}")
    
    data_directory = os.path.join(project_root, 'dataset', data_name, 'labeled_data')
    selected_features = [f'e{i}' for i in range(dof)] 
    all_csv_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)
    results_list = []
    results_csv_path = os.path.join(log_dir, 'hyperparameter_results_expanded.csv')

    if not all_csv_files:
        logging.error(f"No CSV files found in '{data_directory}'. Please check the path.")
    else:
        
        # --- Start Hyperparameter Search Loop ---
        for seq_num in seq_nums:
            # --- Pre-load datasets to avoid re-loading in the inner loop ---
            logging.info("--- Pre-loading all datasets ---")
            with Pool(processes=max(1, cpu_count() // 2)) as pool:
                val_tasks = [(file, 1 if 'no_contact' not in file else 0, selected_features, 'val', seq_num, VALIDATION_GAP) for file in all_csv_files]
                all_val_datasets = pool.map(load_dataset_worker, val_tasks)
                val_dataset = ConcatDataset([d for d in all_val_datasets if d is not None and len(d) > 0])
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)
                logging.info(f"Validation dataset loaded with {len(val_dataset)} samples.")

                train_tasks = [(file, 1 if 'no_contact' not in file else 0, selected_features, 'train', seq_num, gaps[0]) for file in all_csv_files]
                all_train_datasets = pool.map(load_dataset_worker, train_tasks)
                train_dataset = ConcatDataset([d for d in all_train_datasets if d is not None and len(d) > 0])
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True)
                logging.info(f"Training dataset loaded with {len(train_dataset)} samples.")

            for d_model in d_models:
                for n_head in n_heads:
                    if d_model % n_head != 0:
                        continue
                    for num_layers in num_encoder_layers_list:
                        for dropout in dropout_rates:
                            for lr in learning_rates:
                                for wd in weight_decays:
                                    logging.info(f"--- Starting Training ---")
                                    logging.info(f"PARAMS: num_features={seq_num}, d_model={d_model}, n_head={n_head}, num_layers={num_layers}, dropout={dropout}, lr={lr}, wd={wd}")
                                    
                                    model = TransformerModel(
                                        num_features=seq_num, 
                                        d_model=d_model, 
                                        nhead=n_head, 
                                        num_encoder_layers=num_layers,
                                        dim_feedforward=d_model*4,
                                        dropout=dropout # Pass dropout to the model
                                    )
                                    
                                    model_name = f'dModel{d_model}_nHead{n_head}_nLayer{num_layers}_sNum{seq_num}s'
                                    model_path = os.path.join(log_dir, f'{model_name}.pth')

                                    _, accuracy = train_model(
                                        train_loader=train_loader, val_loader=val_loader,
                                        model=model, n_epochs=n_epochs,
                                        learning_rate=lr, weight_decay=wd,
                                        model_path=model_path
                                    )
                                    
                                    final_model_path = os.path.join(log_dir, f'{model_name}_acc{accuracy:.2f}.pth')
                                    if os.path.exists(model_path):
                                        os.rename(model_path, final_model_path)
                                    
                                    results_list.append({
                                        'seq_num':seq_num, 'd_model': d_model, 'n_head': n_head, 'num_layer': num_layers,
                                        'dropout': dropout, 'learning_rate': lr, 'weight_decay': wd,
                                        'ACC': accuracy
                                    })
                                    results_df = pd.DataFrame(results_list)
                                    # Check if the file already exists
                                    results_df.to_csv(results_csv_path, index=False)
                                    logging.info(f"Results updated at {results_csv_path}")
                                    
    logging.info("\n--- Hyperparameter search complete ---")