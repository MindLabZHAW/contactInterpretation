import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import os
import glob
from torch.utils.data import DataLoader, ConcatDataset, Subset
from multiprocessing import Pool, cpu_count

# --- Add Project Root to Python Path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    from src.data import LoadSeqDataset
    from src.training import Trainer
    import src.models as model_zoo

except ImportError as e:
    print(f"Error: Could not import necessary modules. Please ensure the project structure is correct.")
    print(f"Details: {e}")
    sys.exit(1)

def setup_logging():
    """Configures a basic logger for clean console output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def load_dataset_worker(args):
    """Worker function for parallel data loading."""
    file_path, label, selected_features, mode, window_length, gap = args
    return LoadSeqDataset(file_path, label, selected_features, mode, window_length, gap)

def main():
    """
    Main entry point for the training script.
    """
    parser = argparse.ArgumentParser(
        description="Run the training pipeline for contact detection and localization models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration YAML file (e.g., config/cnnBiLSTMFrankaMain.yaml).'
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    config_path = project_root / args.config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from: {config_path}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)

    # --- Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- Data Loading Setup ---
    data_directory = config['project']['dataset_dir'].format(data_name=config['project']['data_name'])
    all_csv_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)
    selected_features = [f'e{i}' for i in range(config['project']['dof'])]

    # --- Loop through both detection and localization models ---
    for model_key in ['detection_model',  'localization_model']: #
        logging.info(f"\n--- Starting Training for {model_key} ---")
        
        model_cfg = config[model_key]
        trainer_params = model_cfg['trainer_params']
        
        # --- Model Initialization ---
        models_dir = config['project']['models_dir'].format(data_name=config['project']['data_name'])
        weights_template = model_cfg['weights_file_template']
        format_context = {**model_cfg.get('model_init_args', {}), **model_cfg.get('filename_params', {})}
        final_weights_file = weights_template.format(**format_context)
        models_dir = f'{models_dir}/{model_cfg["type"]}/{model_cfg["version"]}/{final_weights_file}'
        print(models_dir)

        try:
            ModelClass = getattr(model_zoo, model_cfg['architecture'])
            model = ModelClass(**model_cfg['model_init_args'])
        except Exception as e:
            raise TypeError(f"Error instantiating '{model_cfg['architecture']}'. Check 'model_init_args'. Error: {e}")
        
        try:
            model.load_state_dict(torch.load(models_dir, map_location=device))
        except FileNotFoundError:
            logging.error(f"Model weights file not found at: {models_dir}")
            raise

        
        # --- Optimizer, Scheduler, and Loss (Task-Specific) ---
        if trainer_params['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=trainer_params['learning_rate'], weight_decay=1e-5)
        else:
            optimizer = optim.Adam(model.parameters(), lr=trainer_params['learning_rate'], weight_decay=1e-5)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=trainer_params['early_stop_patience'])
        
        if model.task == 'localization':
            criterion = nn.CrossEntropyLoss()
        else: # detection
            criterion = nn.BCEWithLogitsLoss()

        # --- Parallel Data Loading ---
        logging.info("Starting parallel data loading...")
        with Pool(processes=max(1, cpu_count() // 2)) as pool:
            train_tasks = []
            val_tasks = []
            for file_path in all_csv_files:
                if model_key == 'detection_model':
                    label = 1 if 'no_contact' not in file_path else 0
                else: # localization_model
                    label = next((num for name, num in config['labels'].items() if name in file_path), 0)

                train_tasks.append((
                    file_path, torch.tensor(label), selected_features, 'train', 
                    trainer_params['window_length'], trainer_params['training_gap']
                ))
                val_tasks.append((
                    file_path, torch.tensor(label), selected_features, 'val', 
                    trainer_params['window_length'], trainer_params['validation_gap']
                ))

            all_train_datasets = pool.map(load_dataset_worker, train_tasks)
            train_dataset_master = ConcatDataset([d for d in all_train_datasets if len(d) > 0])

            all_val_datasets = pool.map(load_dataset_worker, val_tasks)
            val_dataset_master = ConcatDataset([d for d in all_val_datasets if len(d) > 0])

        # --- Task-Specific Dataset Filtering ---
        if model.task == 'localization':
            logging.info("Filtering for localization task (contact data only)...")
            train_collision_indices = [i for i, (_, label) in enumerate(train_dataset_master) if label > 0]
            train_dataset = Subset(train_dataset_master, train_collision_indices)
            val_collision_indices = [i for i, (_, label) in enumerate(val_dataset_master) if label > 0]
            val_dataset = Subset(val_dataset_master, val_collision_indices)
        else:
            train_dataset = train_dataset_master
            val_dataset = val_dataset_master

        logging.info(f"Training dataset size: {len(train_dataset)} samples.")
        logging.info(f"Validation dataset size: {len(val_dataset)} samples.")
        
        train_loader = DataLoader(train_dataset, batch_size=trainer_params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=trainer_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        # --- Trainer Initialization and Execution ---
        trainer = Trainer(model, optimizer, criterion, device, scheduler=scheduler)
        
        # --- Define Model Save Path ---
        save_dir = os.path.join(config['project']['save_model_dir'].format(data_name=config['project']['data_name']), model_cfg['type'], str(trainer_params['batch_size']))
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, 'best_model.pth')
        trainer.fit(
            train_loader, val_loader, 
            n_epochs=trainer_params['epochs'], 
            model_save_path=model_save_path,
            final_model_name=final_weights_file,
            early_stop_patience=trainer_params['early_stop_patience']
        )
        logging.info(f"--- Finished Training for {model_key} ---")

if __name__ == '__main__':
    setup_logging()
    main()

    