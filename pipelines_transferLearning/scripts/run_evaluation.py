import argparse
import logging
import sys
from pathlib import Path
import yaml

# --- Add Project Root to Python Path ---
# This allows the script to find and import modules from the 'src' directory.
project_root = Path(__file__).resolve().parents[1]
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
    config_path = project_root / args.config
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