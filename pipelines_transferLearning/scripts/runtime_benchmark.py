import argparse
import logging
import sys
import os
import yaml
import torch
import time, re, glob
from pathlib import Path
import torch.nn as nn

# --- Add Project Root to Python Path ---
# This allows the script to find and import modules from the 'src' directory.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
project_root_run= Path(__file__).resolve().parents[2]

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
    
    #logging.info(f"--- Results for {device} ---")
    #logging.info(f"Total time for {num_runs} runs: {total_time:.4f} seconds")
    logging.info(f"Average time per batch (batch_size={batch_size}): {avg_time_per_batch * 1000:.4f} ms")
    #logging.info(f"Average inference time per sample: {avg_time_per_sample * 1000:.4f} ms")
    #logging.info(f"Estimated Real-time FPS: {fps:.2f}\n")

def find_best_accuracy(config: dict, model_key: str):
    """
    Checks if a model's accuracy is 'idk'. If so, scans its directory
    for the file matching all other params with the highest accuracy.
    """
    if model_key not in config:
        logging.warning(f"'{model_key}' not found in config. Skipping accuracy check.")
        return

    model_cfg = config[model_key]
    
    # --- 1. Find the correct parameter dictionary ---
    params_key = None
    if 'filename_params' in model_cfg:
        params_key = 'filename_params'
    else:
        logging.info(f"No 'file_params' or 'filename_params' found for '{model_key}'. Skipping 'idk' check.")
        return
        
    params_dict = model_cfg.get(params_key, {})
    accuracy_value = params_dict.get('accuracy')

    # --- 2. Check if search is needed ---
    if accuracy_value != 'idk':
        #logging.info(f"Accuracy for '{model_key}' is '{accuracy_value}'. No search needed.")
        return

    #logging.info(f"Accuracy for '{model_key}' is 'idk'. Searching for best model...")

    # --- 3. Get the model directory path ---
    try:
        base_models_dir_template = config['project']['models_dir']
        
        if '{data_name}' in base_models_dir_template:
            base_models_dir = base_models_dir_template.format(data_name=config['project']['data_name'])
        else:
            base_models_dir = base_models_dir_template
            
        folder_name_ = model_cfg.get('type', '')
        folder_name = model_cfg.get('version', '')
        
        full_model_dir = os.path.join(project_root_run, base_models_dir, folder_name_, folder_name)
        
        if not os.path.isdir(full_model_dir):
            logging.error(f"Model directory not found at: {full_model_dir}")
            sys.exit(1)
            
    except KeyError as e:
        logging.error(f"Missing expected key in config to find model directory: {e}")
        sys.exit(1)

    # --- 4. Build the search pattern ---
    try:
        # Get all parameters needed for the template
        format_context = {
            **model_cfg.get('model_init_args', {}),
            **model_cfg.get(params_key, {})
        }
        
        # Remove 'accuracy' because it's the key we're searching for
        if 'accuracy' in format_context:
            del format_context['accuracy']

        # Get the template and replace the accuracy format with a wildcard
        weights_template = model_cfg['weights_file_template']
        # This regex finds any {key:format} pattern for 'accuracy'
        glob_template = re.sub(r'\{accuracy[^\}]*\}', '*', weights_template)
        
        # Create the final glob search pattern
        search_pattern = glob_template.format(**format_context)
        
    except KeyError as e:
        logging.error(f"Config is missing a key required by the 'weights_file_template': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error building search pattern: {e}")
        sys.exit(1)

    # --- 5. Scan directory with glob and parse with regex ---
    #logging.info(f"Searching in: {full_model_dir}")
    #logging.info(f"Using pattern: {search_pattern}")
    
    # Regex to extract the accuracy value from the matching filenames
    if model_cfg['architecture'] == 'TransformerModel':
        acc_regex = re.compile(r"_acc(\d+\.\d+)")
    else:
        acc_regex = re.compile(r"_accuracy(\d+\.\d+)")
    
    max_acc = -1.0
    files_found = 0
    
    # Use glob to find all files matching the pattern
    search_path = os.path.join(full_model_dir, search_pattern)
    for filepath in glob.glob(search_path):
        filename = os.path.basename(filepath)
        match = acc_regex.search(filename)
        
        if match:
            files_found += 1
            acc_float = float(match.group(1))
            
            if acc_float > max_acc:
                max_acc = acc_float

    # --- 6. Update the config ---
    if max_acc > -1.0:
        #logging.info(f"Found {files_found} matching models. Best accuracy: {max_acc:.4f}.")
        # --- THE FIX ---
        # Update the config with the FLOAT, not the string, using the correct key
        config[model_key][params_key]['accuracy'] = max_acc
    else:
        logging.error(f"Accuracy for '{model_key}' was 'idk', but no models matching "
                      f"the pattern '{search_pattern}' were found in {full_model_dir}.")
        sys.exit(1)

def run_benchmark(batch_size, model, task):
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
        default=f'config/{model}',
        help='Path to the configuration YAML file (e.g., config/fineTuningCNNBiLSTM2FrankaMainTOFrankaMindlab.yaml).'
    )
    parser.add_argument(
        '--model_key',
        type=str,
        default=task,
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
        #logging.info(f"Successfully loaded configuration from: {config_path}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    
    dof = config['project']['dof']

    find_best_accuracy(config, 'detection_model')
    find_best_accuracy(config, 'localization_model')

    # --- 1. Benchmark on CPU ---
    #logging.info("--- Preparing CPU Benchmark ---")
    cpu_device = torch.device("cpu")
    try:
        cpu_model, model_cfg = load_model_from_config(config, args.model_key, cpu_device)
        benchmark(cpu_model, model_cfg, cpu_device, dof, args.batch_size, args.num_runs)
    except Exception as e:
        logging.error(f"Failed to benchmark on CPU: {e}", exc_info=True)

    # --- 2. Benchmark on GPU ---
    if torch.cuda.is_available():
        #logging.info("--- Preparing GPU Benchmark ---")
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
    models= ['_cnnBiLSTM1FrankaMain.yaml', '_cnnBiLSTM2FrankaMindlab.yaml', '_cnnBiLSTM3UR5.yaml',
            '_Transformer1FrankaMainBest.yaml', '_Transformer2FrankaMainLight.yaml', 
            '_Transformer3FrankaMindlab.yaml', '_Transformer4UR5.yaml']
    
    models= ['_cnnBiLSTM1FrankaMain.yaml', '_cnnBiLSTM2FrankaMindlab.yaml', '_cnnBiLSTM3UR5.yaml',
            '_Transformer1FrankaMain.yaml', 
            '_Transformer3FrankaMindlab.yaml', '_Transformer4UR5.yaml']    
    
    tasks= ['detection_model', 'localization_model']
    
    for task in tasks:
        for model in models:
            for batch_size in [1, 100]:
                run_benchmark(model=f'fastestModels/{model}', task=task, batch_size=batch_size)