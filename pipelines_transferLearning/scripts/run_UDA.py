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
from torch.utils.data import DataLoader, ConcatDataset
from multiprocessing import Pool, cpu_count

# --- Add Project Root to Python Path ---
# This ensures that the script can be run from anywhere and still find the 'src' module.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    from src.data import LoadSeqDataset
    from src.training import AdversarialTrainer # Your DANN trainer
    import src.models as model_zoo
except ImportError as e:
    print(f"Error: Could not import necessary modules from 'src'.")
    print(f"Project root detected as: {project_root}")
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
    """
    Worker function for parallel data loading. Unpacks arguments and creates a LoadSeqDataset.
    """
    file_path, label, selected_features, mode, window_length, gap = args
    dataset = LoadSeqDataset(file_path, label, selected_features, mode, window_length, gap)
    # Return None if the dataset is empty to filter it out later
    return dataset if len(dataset) > 0 else None

def create_dataloader_from_dir(directory: str, mode: str, task: str, config: dict, domain: str):
    """
    Creates a master DataLoader by loading and concatenating datasets from a directory in parallel.
    """
    #project_cfg = config['project']
    task_cfg = config[f'{task}_model']
    trainer_params = task_cfg['trainer_params']
    
    all_csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")

    # Use a simple feature naming convention, e.g., ['e0', 'e1', ..., 'e6']
    if domain == 'source':
        dof = config['project']['dof']
    else:
        dof = config['transfer_learning']['dof']
    
    selected_features = [f'e{i}' for i in range(dof)]
    
    tasks = []
    for file_path in all_csv_files:
        # Infer label based on filename and task
        if task == 'detection':
            label = 0 if 'no_contact' in file_path else 1
        else: # localization
            label = next((num for name, num in config['labels'].items() if name in file_path), 0)
        
        gap = trainer_params.get(f'{mode}_gap', trainer_params['training_gap'])
        tasks.append((
            file_path, torch.tensor(label), selected_features, mode,
            trainer_params['window_length'], gap
        ))

    # Use multiprocessing to load datasets in parallel
    logging.info(f"Starting parallel data loading for {len(tasks)} files...")
    with Pool(processes=max(1, cpu_count() // 2)) as pool:
        all_datasets = pool.map(load_dataset_worker, tasks)
    
    # Filter out any empty datasets that might have been created
    valid_datasets = [d for d in all_datasets if d is not None]
    if not valid_datasets:
        return None

    master_dataset = ConcatDataset(valid_datasets)
    
    logging.info(f"Created '{mode}' dataset for '{task}' with {len(master_dataset)} total samples.")
    
    return DataLoader(
        master_dataset,
        batch_size=trainer_params['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=4,
        pin_memory=True
    )

def main():
    parser = argparse.ArgumentParser(
        description="Run the Adversarial Domain Adaptation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--task', type=str, required=True, choices=['detection', 'localization'], help='The task to train.')
    args = parser.parse_args()

    # --- Load Configuration ---
    config_path = project_root / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info(f"Successfully loaded configuration from: {config_path}")

    # --- Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    project_cfg = config['project']
    tl_cfg = config['transfer_learning']
    model_cfg = config[f'{args.task}_model']
    trainer_params = model_cfg['trainer_params']

    # --- Data Loading ---
    source_dir = project_cfg['dataset_dir'].format(data_name=project_cfg['data_name'])
    target_dir = tl_cfg['target_dataset_dir']
    
    source_loader = create_dataloader_from_dir(source_dir, 'train', args.task, config, domain= 'source')
    val_loader = create_dataloader_from_dir(source_dir, 'val', args.task, config, domain= 'source')
    target_loader = create_dataloader_from_dir(target_dir, 'train', args.task, config, domain= 'target')

    if not all([source_loader, val_loader, target_loader]):
        logging.error("Failed to create one or more data loaders. Exiting.")
        sys.exit(1)

    # --- Model Initialization ---
    ModelClass = getattr(model_zoo, model_cfg['architecture'])
    num_classes = 1 if args.task == 'detection' else project_cfg['dof']
    model = ModelClass(**model_cfg['model_init_args'], domain_discriminator = True)
    
    # --- Optimizer and Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=trainer_params['learning_rate'], weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss() if args.task == 'detection' else nn.CrossEntropyLoss()

    # --- Define Model Save Path ---
    save_dir = project_cfg['save_model_dir'].format(data_name=project_cfg['data_name'])
    model_save_path = os.path.join(save_dir, f"DANN_adapted_{args.task}_best_model.pth")
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Trainer Initialization and Execution ---
    trainer = AdversarialTrainer(
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        task_criterion=criterion,
        device=device,
        model_save_path=model_save_path,
        use_dynamic_lambda=tl_cfg.get('use_dynamic_lambda', True)
    )
    
    logging.info(f"--- Starting Adversarial Training for {args.task.upper()} ---")
    trainer.train(epochs=trainer_params['epochs'])
    logging.info(f"--- Finished Training for {args.task.upper()} ---")

if __name__ == '__main__':
    setup_logging()
    main()