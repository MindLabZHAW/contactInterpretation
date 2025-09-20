import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
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

from pipelines.src.dataset_loader import LoadSeqDataset
from pipelines.models.cnnLSTM_contactDetection import cnnLSTM

# Helper function for parallel data loading
def load_dataset_worker(args):
    file_path, label, selected_features, seq_num, gap = args
    return LoadSeqDataset(file_path, label, selected_features, seq_num, gap)

def train_model(full_dataset,model,  n_epochs=50, batch_size=64, learning_rate=0.001, model_path='cnn_lstm_model'):
    # This function remains the same as your provided script
    # ... (training and validation loop) ...
    train_size = int(0.5 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=min(4, os.cpu_count()), pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=min(4, os.cpu_count()), pin_memory=True)
    
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
    # --- Configuration ---
    project_root = os.getcwd().replace('pipelines','')

    data_name = 'franka_main'
    dof = 7

    # --- Hyperparameter Search Space ---
    hidden_sizes = [32, 64, 128, 256]
    num_layers_list = [1, 2, 3]
    seq_nums = [30, 50, 80, 100, 150, 200]
    gap = 5#[3, 5, 10, 15]

    gap = 5
    batch_size = 72
    n_epochs = 40

    log_dir = f'{project_root}/pipelines/trained_models/{data_name}/contact_detection_v2/{batch_size}/'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'training_log_{time.time()}.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    
    logging.info(f"Logging to {log_file}")

    # --- Setup Logging and Reproducibility ---
    # ... (logging and seeding is unchanged) ...

    # --- Optimized Data Loading from Labeled CSVs ---
    data_directory = os.path.join(project_root, 'dataset', data_name, 'labeled_data')
    dict_label = {'link7': 7, 'link6': 6, 'link5': 5, 'link4': 4, 'link3': 3, 'link2': 2, 'link1': 1, 'no_contact': 0}
    selected_features = [f'e{i}' for i in range(dof)]
    
    all_csv_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)
    results_list = []
    results_csv_path = os.path.join(log_dir, 'hyperparameter_results.csv')

    if not all_csv_files:
        logging.error(f"No CSV files found in '{data_directory}'. Please check the path.")
    else:
        for seq_num in seq_nums:
            for hidden_size in hidden_sizes:
                for num_layers in num_layers_list:
                    logging.info(f"Starting data loading for seq_num={seq_num}, gap={gap}, hidden_size={hidden_size}, num_layers={num_layers}")
                    # Create a list of arguments for the parallel worker function
                    tasks = []
                    for file in all_csv_files:
                        label = 1 if 'no_contact' not in file else 0
                        tasks.append((file, torch.tensor(label), selected_features, seq_num, gap))

                    # Use a multiprocessing Pool to load datasets in parallel
                    logging.info(f"Starting parallel data loading with {cpu_count()} workers...")
                    with Pool(processes=cpu_count()) as pool:
                        all_datasets = pool.map(load_dataset_worker, tasks)
                    
                    # Combine them into a single dataset
                    master_dataset = ConcatDataset(all_datasets)
                    logging.info(f"Successfully loaded and combined data from {len(all_csv_files)} files into a dataset with {len(master_dataset)} samples.")
                    
                    # --- Train the Model ---
                    model = cnnLSTM(num_features_joints=seq_num, hidden_size=hidden_size, num_layers=num_layers)
                    model_name = f'numLayer{num_layers}_hiddenSize{hidden_size}_seq_num{seq_num}_gap{gap}'

                    trained_model, accuracy  = train_model(full_dataset=master_dataset,model=model, n_epochs=n_epochs,batch_size=batch_size, model_path = f'{log_dir}{model_name}.pth')
                    os.rename(f'{log_dir}{model_name}.pth', f'{log_dir}{model_name}_accuracy{accuracy:.2f}.pth')

                    results_list.append({
                        'gap': gap,
                        'seq_num': seq_num,
                        'hidden_size': hidden_size,
                        'num_layer': num_layers,
                        'ACC': accuracy
                    })
                    results_df = pd.DataFrame(results_list)
                    results_df.to_csv(results_csv_path, index=False)
                    logging.info(f"Results saved to {results_csv_path}")
    logging.info("\n--- Hyperparameter search complete ---")