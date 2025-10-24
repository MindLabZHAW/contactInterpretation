import argparse
import logging
import sys
from pathlib import Path
import yaml, re, os, glob

# --- Add Project Root to Python Path ---
# This allows the script to find and import modules from the 'src' directory.
project_root = Path(__file__).resolve().parents[1]
project_root_run= Path(__file__).resolve().parents[2]

sys.path.insert(0, str(project_root))

try:
    # --- Import Core Components from the New Structure ---
    from src.evaluation import Evaluator
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
        logging.info(f"Accuracy for '{model_key}' is '{accuracy_value}'. No search needed.")
        return

    logging.info(f"Accuracy for '{model_key}' is 'idk'. Searching for best model...")

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
    logging.info(f"Searching in: {full_model_dir}")
    logging.info(f"Using pattern: {search_pattern}")
    
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
        logging.info(f"Found {files_found} matching models. Best accuracy: {max_acc:.4f}.")
        # --- THE FIX ---
        # Update the config with the FLOAT, not the string, using the correct key
        config[model_key][params_key]['accuracy'] = max_acc
    else:
        logging.error(f"Accuracy for '{model_key}' was 'idk', but no models matching "
                      f"the pattern '{search_pattern}' were found in {full_model_dir}.")
        sys.exit(1)

def main():
    """
    Main entry point for the evaluation script.

    This script orchestrates the evaluation process by:
    1. Parsing the path to a configuration file from the command line.
    2. Loading the specified configuration.
    3. Initializing the main `Evaluator` class with the configuration.
    4. Starting the evaluation run.
    """
    parser = argparse.ArgumentParser(
        description="Run the two-stage contact detection and localization pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration YAML file (e.g., config/_cnnBiLSTMFrankaMain.yaml).'
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    #config_path = project_root / args.config
    config_path = f'{project_root}/config/{args.config}'

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from: {config_path}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)

    find_best_accuracy(config, 'detection_model')
    find_best_accuracy(config, 'localization_model')

    # --- Initialize and Run the Pipeline ---
    try:
        # Pass the loaded config dictionary to the Evaluator
        pipeline = Evaluator(config)
        pipeline.run()
        logging.info("Evaluation pipeline finished successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the evaluation run: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    setup_logging()
    main()