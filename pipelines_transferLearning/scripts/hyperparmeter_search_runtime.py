import argparse
import logging
import sys
import os
import torch
import time
import itertools
from pathlib import Path
import torch.nn as nn
import pandas as pd

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
        format='%(asctime)s - %(message)s', # Simpler format for clean table
        stream=sys.stdout
    )

def instantiate_model(architecture_name: str, init_args: dict, device: torch.device) -> tuple:
    """
    Dynamically instantiates a model based on architecture name and init args.
    Does NOT load any weights.
    """
    try:
        ModelClass = getattr(model_zoo, architecture_name)
        model = ModelClass(**init_args)
        model.to(device).eval() # Set to evaluation mode (disables dropout, etc.)
    except Exception as e:
        # Provide more context on error
        logging.error(f"Error instantiating '{architecture_name}' with args {init_args}.")
        logging.error(f"Ensure all keys in init_args match the model's __init__ signature.")
        logging.error(f"Original Error: {e}", exc_info=True)
        raise TypeError(f"Error instantiating '{architecture_name}'.")

    # Create a mock 'model_cfg' to pass to the benchmark function
    # This ensures compatibility with the benchmark function's expectation
    model_cfg = {
        'architecture': architecture_name,
        'model_init_args': init_args
    }
    
    return model, model_cfg

def benchmark(model: nn.Module, model_cfg: dict, device: torch.device, dof: int, batch_size: int, num_runs: int) -> float:
    """
    Runs the benchmark for a given model and device.
    Returns the average time per batch in seconds.
    """
    try:
        # 'num_features' is our name for Window Size
        window_length = model_cfg['model_init_args']['num_features']
    except KeyError:
        logging.error("Config Error: Could not find 'num_features' in 'model_init_args'.")
        return -1.0

    # Expected shape is (batch, dof, seq_len)
    input_shape = (batch_size, dof, window_length)
    random_input = torch.rand(input_shape, device=device)
    
    # Warm-up runs
    try:
        with torch.no_grad():
            for _ in range(20):
                _ = model(random_input)
    except Exception as e:
        logging.error(f"Error during model warm-up/inference: {e}")
        logging.error(f"Model: {model_cfg['architecture']}, Shape: {tuple(input_shape)}, Config: {model_cfg['model_init_args']}")
        return -1.0

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
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_runs
    
    return avg_time_per_batch

if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Benchmark model inference speed across a hyperparameter search space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dof',
        type=int,
        required=True,
        help='Degrees of Freedom for the robot (e.g., 6 or 7). This is required for model instantiation.'
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=100, 
        help='Number of inference runs to average over for each configuration.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.csv', 
        help='Path to save the CSV results file.'
    )
    args = parser.parse_args()

    # --- Define Hyperparameter Search Spaces ---
    
    # --- NEW: Define all four search spaces ---
    bilstm_space = {
        'num_features': [30, 50, 80, 100, 150, 200],
        'hidden_size': [32, 64, 128, 256],
        'num_layers': [1, 2, 3],
        'task': ['detection']
    }
    bilstm_space_localization = {
        'num_features': [30, 50, 80, 100, 150, 200, 250, 300],
        'hidden_size': [32, 64, 128, 256],
        'num_layers': [1, 2, 3],
        'task': ['localization']
    }
    transformer_space = {
        'num_features': [100, 200, 300],
        'd_model': [64, 128, 256, 512],
        'num_encoder_layers': [1, 4, 8],
        'nhead': [1, 4, 8],
        'task': ['detection']
    }
    transformer_space_localization = {
        'num_features': [100, 200, 300],
        'd_model': [128, 256, 512],
        'num_encoder_layers': [1, 4, 8],
        'nhead': [1, 4, 8],
        'task': ['localization']
    }
    # --- End of new section ---


    # --- MODIFIED: List of models to test ---
    # We now list all four combinations.
    model_spaces = [
        ('cnnBiLSTM', bilstm_space),
        ('cnnBiLSTM', bilstm_space_localization),
        ('TransformerModel', transformer_space),
        ('TransformerModel', transformer_space_localization)
    ]

    # --- Start Benchmark ---
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0") if torch.cuda.is_available() else None
    
    results_list = [] 

    logging.info(f"Starting hyperparameter search for inference speed...")
    logging.info(f"Using DoF: {args.dof}, Num Runs: {args.num_runs}")
    logging.info(f"Results will be saved to: {args.output}") 
    logging.info("-" * 80)
    logging.info(f"{'Model':<16} | {'Task':<12} | {'Batch':<5} | {'CPU (ms)':<10} | {'GPU (ms)':<10} | {'Hyperparameters'}")
    logging.info("-" * 80)

    for batch_size in [1, 100]:
        for arch_name, space in model_spaces:
            
            # Get all combinations of hyperparameters
            keys = space.keys()
            value_combos = itertools.product(*space.values())

            for v_combo in value_combos:
                # init_args will now correctly contain 'task'
                init_args = dict(zip(keys, v_combo))

                # Special check for Transformers: d_model must be divisible by n_head
                if 'd_model' in init_args and 'nhead' in init_args:
                    if init_args['d_model'] % init_args['nhead'] != 0:
                        logging.warning(f"Skipping invalid config: d_model={init_args['d_model']} not divisible by nhead={init_args['nhead']}")
                        continue
                
                # params_str will now automatically include the task
                params_str = ", ".join(f"{k}={v}" for k, v in init_args.items())

                # --- 1. Benchmark on CPU ---
                try:
                    cpu_model, model_cfg = instantiate_model(arch_name, init_args, cpu_device)
                    cpu_time_s = benchmark(cpu_model, model_cfg, cpu_device, args.dof, batch_size, args.num_runs)
                    cpu_time_ms = cpu_time_s * 1000
                    del cpu_model # Free memory
                except Exception as e:
                    logging.error(f"Failed benchmark for {arch_name} on CPU. Config: {params_str}")
                    cpu_time_ms = None 
                    continue

                # --- 2. Benchmark on GPU ---
                gpu_time_ms = None  
                gpu_log_str = "N/A" 
                
                if gpu_device:
                    try:
                        gpu_model, model_cfg = instantiate_model(arch_name, init_args, gpu_device)
                        gpu_time_s = benchmark(gpu_model, model_cfg, gpu_device, args.dof, batch_size, args.num_runs)
                        gpu_time_ms = gpu_time_s * 1000  
                        gpu_log_str = f"{gpu_time_ms:<10.4f}" 
                        del gpu_model # Free memory
                    except Exception as e:
                        logging.error(f"Failed benchmark for {arch_name} on GPU. Config: {params_str}")
                        gpu_log_str = "Error" 
                        gpu_time_ms = None    
                
                # Log the result
                logging.info(f"{arch_name:<16} | {init_args.get('task', 'N/A'):<12} | {batch_size:<5} | {cpu_time_ms:<10.4f} | {gpu_log_str} | {params_str}")

                # Store data for CSV
                row_data = {
                    'Model': arch_name,
                    'Batch_Size': batch_size,
                    'CPU_Time_ms': cpu_time_ms,
                    'GPU_Time_ms': gpu_time_ms,
                }
                # Unpack all hyperparameters (including task) into their own columns
                row_data.update(init_args)
                results_list.append(row_data)
                
    logging.info("-" * 80)
    logging.info("Benchmark search complete.")

    # --- Save results to CSV ---
    try:
        logging.info(f"Saving results to {args.output}...")
        df = pd.DataFrame(results_list)
        
        # Re-order columns to be more logical (optional but clean)
        base_cols = ['Model', 'task', 'Batch_Size', 'CPU_Time_ms', 'GPU_Time_ms']
        
        # Get all OTHER hyperparameter columns
        param_cols = [col for col in df.columns if col not in base_cols]
        df = df[base_cols + sorted(param_cols)]
        
        df.to_csv(args.output, index=False)
        logging.info(f"Successfully saved results to {args.output}")
    except Exception as e:
        logging.error(f"Failed to save CSV file: {e}")