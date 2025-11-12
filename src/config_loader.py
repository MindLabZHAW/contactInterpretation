import json
import yaml  # <-- Added YAML support
import logging
from typing import Dict 

class ConfigLoader:
    """
    A utility class for loading and parsing data from JSON and YAML files.
    """
    def load_json(self, file_path: str) -> Dict: 
        """
        Loads data from a specified JSON file.
        """
        logging.info(f"Loading JSON from '{file_path}'...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logging.info(f"✅ Successfully loaded '{file_path}'.")
            return data
        except FileNotFoundError:
            logging.error(f"File not found at path: {file_path}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Could not parse JSON file. Check for syntax errors in {file_path}")
            return None

    def load_yaml(self, file_path: str) -> Dict:
        """
        Loads data from a specified YAML file.
        """
        logging.info(f"Loading YAML from '{file_path}'...")
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            logging.info(f"✅ Successfully loaded '{file_path}'.")
            return data
        except FileNotFoundError:
            logging.error(f"File not found at path: {file_path}")
            return None
        except yaml.YAMLError as e:
            logging.error(f"Could not parse YAML file: {e}")
            return None