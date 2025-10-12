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
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipelines.src_exp2.dataset_loader import LoadSeqDataset
from pipelines.models.cnnLSTM_contactDetection import cnnLSTM

# Helper function for parallel data loading (unchanged)
def load_dataset_worker(args):
    """Worker function for parallel data loading."""
    file_path, label, selected_features, mode, seq_num, gap = args
    return LoadSeqDataset(file_path, label, selected_features, mode, seq_num, gap)

# train_model function (unchanged)
def train_model(train_loader, val_loader,model,  n_epochs=50, batch_size=64, learning_rate=0.001, model_path='cnn_lstm_model'):
    # This function remains the same as your provided script
    # ... (training and validation loop) ...
    
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
            
            optimizer.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
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
        if current_lr < 0.0005:
            logging.info(f"Learning rate ({current_lr:.6f}) has dropped below the threshold. Stopping training early.")
            break
        
            
    return model, best_val_accuracy


if __name__ == '__main__':
    # --- Configuration (unchanged) ---
    project_root = os.getcwd().replace('pipelines','')
    data_name = 'franka_main'
    dof = 7
    hidden_sizes = [32, 64, 128, 256]#, 512, 1024]
    num_layers_list = [1, 2, 3]
    seq_nums = [30, 50, 80, 100, 150, 200]#, 250, 300]
    gaps = [3, 5, 10, 15]
    batch_size = 64
    n_epochs = 40

    VALIDATION_GAP = 5
    
    log_dir = f'{project_root}/pipelines/trained_models/{data_name}/contact_detection_v3/{batch_size}/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{time.time()}.txt')
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

            # BUG FIX: Create the process pool ONCE per seq_num to avoid conflicts.
            with Pool(processes=cpu_count()) as pool:
                # 1. Load the VALIDATION dataset using the single, stable pool.
                val_tasks = []
                for file in all_csv_files:
                    label = 1 if 'no_contact' not in file else 0
                    val_tasks.append((file, torch.tensor(label), selected_features, 'val', seq_num, VALIDATION_GAP))
                
                logging.info(f"Loading validation datasets with fixed gap={VALIDATION_GAP}...")
                all_val_datasets = pool.map(load_dataset_worker, val_tasks)
                
                val_dataset = ConcatDataset(all_val_datasets)
                val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                          num_workers=min(4, os.cpu_count()), pin_memory=True, persistent_workers=True)
                logging.info(f"Created stable Validation Dataset for seq_num={seq_num} with {len(val_dataset)} samples.")

                # 2. Loop through training gaps, REUSING the same pool.
                for train_gap in gaps:
                    logging.info(f"--- Processing for Training Gap = {train_gap} ---")
                    
                    train_tasks = []
                    for file in all_csv_files:
                        label = 1 if 'no_contact' not in file else 0
                        train_tasks.append((file, torch.tensor(label), selected_features, 'train', seq_num, train_gap))

                    logging.info(f"Loading training datasets with gap={train_gap}...")
                    # This uses the same pool created outside the loop
                    all_train_datasets = pool.map(load_dataset_worker, train_tasks)

                    train_dataset = ConcatDataset(all_train_datasets)
                    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=min(4, os.cpu_count()), pin_memory=True, persistent_workers=True)
                    logging.info(f"Created Training Dataset for gap={train_gap} with {len(train_dataset)} samples.")

                    # 3. Loop through model architectures and train.
                    for hidden_size in hidden_sizes:
                        for num_layers in num_layers_list:
                            logging.info(f"Starting training for: seq_num={seq_num}, gap={train_gap}, hidden_size={hidden_size}, num_layers={num_layers}")
                            
                            # Using your specified model initialization
                            model = cnnLSTM(num_features_joints=seq_num, hidden_size=hidden_size, num_layers=num_layers)
                            
                            model_name = f'numLayer{num_layers}_hiddenSize{hidden_size}_seq_num{seq_num}_gap{train_gap}'

                            trained_model, accuracy = train_model(train_loader=train_loader, val_loader=val_loader,
                                                                  model=model, n_epochs=n_epochs,
                                                                  model_path=f'{log_dir}{model_name}.pth')
                            
                            final_model_path = f'{log_dir}{model_name}_accuracy{accuracy:.2f}.pth'
                            if os.path.exists(f'{log_dir}{model_name}.pth'):
                                 os.rename(f'{log_dir}{model_name}.pth', final_model_path)
                            
                            results_list.append({
                                'gap': train_gap,
                                'seq_num': seq_num,
                                'hidden_size': hidden_size,
                                'num_layer': num_layers,
                                'ACC': accuracy,
                                'train_size': len(train_dataset),
                                'val_size': len(val_dataset)
                            })
                            results_df = pd.DataFrame(results_list)
                            results_df.to_csv(results_csv_path, index=False)
                            logging.info(f"Results updated at {results_csv_path}")
                            
    logging.info("\n--- Hyperparameter search complete ---")