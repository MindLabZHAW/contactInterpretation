import argparse
import logging
import sys
import os
import yaml
import torch
import time
from pathlib import Path
import torch.nn as nn

# --- Add Project Root to Python Path ---
# This allows the script to find and import modules from the 'src' directory.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    # --- Import Core Components ---
    # This import is necessary for the getattr() call during model instantiation
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

def load_model_from_config(config: dict, model_key: str, device: torch.device) -> tuple:
    """
    Dynamically loads a model and its weights from the configuration onto a specific device.
    This logic is adapted from your 'evaluator.py' and 'run_fineTuning.py' scripts.
    """
    model_cfg = config[model_key]
    
    # Robustly determine the base models directory
    base_models_dir = config['project']['models_dir']
    if '{data_name}' in base_models_dir:
        base_models_dir = base_models_dir.format(data_name=config['project']['data_name'])
    
    # Format the weights file template
    weights_template = model_cfg['weights_file_template']
    format_context = {**model_cfg.get('model_init_args', {}), **model_cfg.get('filename_params', {})}
    final_weights_file = weights_template.format(**format_context)
    
    # Get subfolders (type and version)
    folder_name_ = model_cfg.get('type', '')
    folder_name = model_cfg.get('version', '')
    
    # Construct the final path
    if folder_name_ and folder_name:
        final_model_path = os.path.join(base_models_dir, folder_name_, folder_name, final_weights_file)
    else:
        # Fallback for configs without type/version (e.g., original trained models)
        final_model_path = os.path.join(base_models_dir, final_weights_file)
        
    logging.info(f"Attempting to load weights from: {final_model_path}")
    
    # Instantiate the model
    try:
        ModelClass = getattr(model_zoo, model_cfg['architecture'])
        model = ModelClass(**model_cfg['model_init_args'])
    except Exception as e:
        raise TypeError(f"Error instantiating '{model_cfg['architecture']}'. Check 'model_init_args'. Error: {e}")
    
    # Load the weights
    try:
        model.load_state_dict(torch.load(final_model_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"Model weights file not found at: {final_model_path}")
        raise
    
    model.to(device).eval()
    logging.info(f"Successfully loaded model '{model_key}' to {device}")
    return model, model_cfg

def benchmark(model: nn.Module, model_cfg: dict, device: torch.device, dof: int, batch_size: int, num_runs: int):
    """
    Runs the benchmark for a given model and device.
    """
    try:
        # 'num_features' is used in the configs to define the window/sequence length
        window_length = model_cfg['model_init_args']['num_features']
    except KeyError:
        logging.error("Config Error: Could not find 'num_features' in 'model_init_args'.")
        return

    # Based on analysis of your dataset_loader, the expected shape is (batch, dof, seq_len)
    input_shape = (batch_size, dof, window_length)
    random_input = torch.rand(input_shape, device=device)
    
    logging.info(f"Device: {device} | Model: {model_cfg['architecture']} | Input Shape: {tuple(input_shape)}")
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(20):
            _ = model(random_input)
    
    # --- Start Timing ---
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter() # Use high-precision timer
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(random_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    # --- End Timing ---
    
    # --- Report Results ---
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_runs
    avg_time_per_sample = avg_time_per_batch / batch_size
    fps = 1.0 / avg_time_per_sample
    
    logging.info(f"--- Results for {device} ---")
    logging.info(f"Total time for {num_runs} runs: {total_time:.4f} seconds")
    logging.info(f"Average time per batch (batch_size={batch_size}): {avg_time_per_batch * 1000:.4f} ms")
    logging.info(f"Average inference time per sample: {avg_time_per_sample * 1000:.4f} ms")
    logging.info(f"Estimated Real-time FPS: {fps:.2f}\n")

def main(batch_size):
    """
    Main entry point for the benchmark script.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark model inference speed on CPU and GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration YAML file (e.g., config/fineTuningCNNBiLSTM2FrankaMainTOFrankaMindlab.yaml).'
    )
    parser.add_argument(
        '--model_key',
        type=str,
        default='detection_model',
        choices=['detection_model', 'localization_model'],
        help='Which model to benchmark from the config file.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=batch_size,
        help='Batch size for inference. Use 1 for real-time latency measurement.'
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=500,
        help='Number of inference runs to average over.'
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
    
    dof = config['project']['dof']

    # --- 1. Benchmark on CPU ---
    logging.info("--- Preparing CPU Benchmark ---")
    cpu_device = torch.device("cpu")
    try:
        cpu_model, model_cfg = load_model_from_config(config, args.model_key, cpu_device)
        benchmark(cpu_model, model_cfg, cpu_device, dof, args.batch_size, args.num_runs)
    except Exception as e:
        logging.error(f"Failed to benchmark on CPU: {e}", exc_info=True)

    # --- 2. Benchmark on GPU ---
    if torch.cuda.is_available():
        logging.info("--- Preparing GPU Benchmark ---")
        gpu_device = torch.device("cuda:0")
        try:
            # Must reload model to move it to the correct device
            gpu_model, model_cfg = load_model_from_config(config, args.model_key, gpu_device)
            benchmark(gpu_model, model_cfg, gpu_device, dof, args.batch_size, args.num_runs)
        except Exception as e:
            logging.error(f"Failed to benchmark on GPU: {e}", exc_info=True)
    else:
        logging.warning("GPU not available. Skipping GPU benchmark.")

if __name__ == '__main__':
    setup_logging()
    main(batch_size=1)