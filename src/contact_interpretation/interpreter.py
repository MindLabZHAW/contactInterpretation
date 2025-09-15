import numpy as np
import os
import re
import torch
import sys
import logging
from typing import Tuple, Optional, Dict, List
from collections import Counter # <-- Added import

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Model Imports ---
try:
    from pipelines.models.cnnLSTM_contactDetection import cnnLSTM as ContactDetectionModel
    from pipelines.models.cnnLSTM_contactLocalization import cnnLSTM as ContactLocalizationModel
except ImportError as e:
    logging.error(f"Fatal: Could not import model definitions from 'pipelines' directory. Error: {e}")
    sys.exit(1)


class ContactAI:
    def __init__(self, num_features: int, majority_voting_window: int = 14):
        self.num_features = num_features
        self.majority_voting_window = majority_voting_window
        # History to store recent predictions for smoothing
        self.detection_history: List[int] = []
        
        # --- Hyperparameter-Based Model Selection ---
        print("\n--- 🧠 Set Hyperparameters for Contact Detection Model ---")
        self.detection_model, self.window_length = self._find_and_load_model_by_hyperparams('contact_detection', ContactDetectionModel)
        
        print("\n--- 🌍 Set Hyperparameters for Contact Localization Model ---")
        self.localization_model, self.loc_window_length = self._find_and_load_model_by_hyperparams('contact_localization', ContactLocalizationModel)

        # --- Final Checks ---
        if self.detection_model is None or self.localization_model is None:
            logging.error("One or both models failed to load. Please check your selections.")
            sys.exit(1)
        else:
            if self.window_length != self.loc_window_length:
                logging.warning(f"Window lengths for detection ({self.window_length}) and localization ({self.loc_window_length}) models differ.")
            self.master_window_length = max(self.window_length, self.loc_window_length)
            self.window = np.zeros((self.num_features, self.master_window_length), dtype=np.float64)

    def _majority_vote(self) -> int:
        """Performs majority voting on the last N predictions."""
        window = self.detection_history[-self.majority_voting_window:]
        most_common = Counter(window).most_common(1)[0][0]
        return most_common

    def predict(self, robot_data: Dict) -> Tuple[int, int, int]:
        """
        Returns raw prediction, smoothed prediction, and localization.
        """
        if self.window is None: return 0, 0, -1
        features = robot_data.get("features")
        if features is None: return 0, 0, -1

        new_column = np.array(features, dtype=np.float64).reshape((self.num_features, 1))
        self.window = np.append(self.window[:, 1:], new_column, axis=1)
        
        with torch.no_grad():
            detection_window = self.window[:, -self.window_length:]
            detection_input = torch.from_numpy(detection_window).unsqueeze(0).double().to(self.device)
            raw_detection_pred = self.detection_model.prediction(detection_input).item()

            # --- Majority Voting Logic ---
            self.detection_history.append(raw_detection_pred)
            smoothed_detection_pred = raw_detection_pred
            if len(self.detection_history) >= self.majority_voting_window:
                smoothed_detection_pred = self._majority_vote()
            # ---------------------------

            localization_pred = -1
            # Use the SMOOTHED prediction to decide whether to run localization
            if smoothed_detection_pred == 1:
                localization_window = self.window[:, -self.loc_window_length:]
                localization_input = torch.from_numpy(localization_window).unsqueeze(0).double().to(self.device)
                localization_pred = self.localization_model.prediction(localization_input).item()
                
        return raw_detection_pred, smoothed_detection_pred, localization_pred

    # ... (the rest of the methods: _get_hyperparameters_from_user, _find_and_load_model_by_hyperparams, _parse_model_name, _load_model remain the same) ...
    def _get_hyperparameters_from_user(self) -> Dict:
        """Prompts the user to enter model hyperparameters."""
        while True:
            try:
                hidden_size = int(input("Enter desired Hidden Size (e.g., 256): "))
                seq_num = int(input("Enter desired Sequence Length (e.g., 100): "))
                return {'hidden_size': hidden_size, 'seq_num': seq_num}
            except ValueError:
                print("Invalid input. Please enter integers only.")

    def _find_and_load_model_by_hyperparams(self, model_type_filter: str, ModelClass: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], int]:
        """Searches for models matching user-defined hyperparameters and prompts for selection if needed."""
        base_dir = os.path.join(project_root, 'pipelines', 'trained_models')
        user_params = self._get_hyperparameters_from_user()
        
        matching_models = []
        for root, _, files in os.walk(base_dir):
            if model_type_filter in root:
                for file in files:
                    if file.endswith('.pth'):
                        parsed_params = self._parse_model_name(file)
                        if parsed_params and \
                           parsed_params['hidden_size'] == user_params['hidden_size'] and \
                           parsed_params['seq_num'] == user_params['seq_num']:
                            
                            display_path = os.path.relpath(os.path.join(root, file), base_dir)
                            full_path = os.path.join(root, file)
                            matching_models.append((display_path, full_path))
        
        if not matching_models:
            logging.error(f"No models found for {model_type_filter} with Hidden Size={user_params['hidden_size']} and Sequence Length={user_params['seq_num']}.")
            return None, -1

        if len(matching_models) == 1:
            print(f"✅ Automatically selected the only matching model: {matching_models[0][0]}")
            full_path = matching_models[0][1]
        else:
            print("\n--- Multiple matching models found ---")
            for i, (display_path, _) in enumerate(matching_models):
                print(f"[{i}] {display_path}")
            try:
                choice = int(input("Select by number: "))
                full_path = matching_models[choice][1]
            except (ValueError, IndexError):
                logging.error("Invalid selection.")
                return None, -1

        return self._load_model(full_path, ModelClass)


    def _parse_model_name(self, filename: str) -> Optional[Dict]:
        params = {}
        try:
            hidden_size_match = re.search(r'hiddenSize(\d+)', filename)
            seq_num_match = re.search(r'seq_num(\d+)', filename)
            if not all([hidden_size_match, seq_num_match]): return None
            num_layers_match = re.search(r'numLayer(\d+)', filename)
            params['num_layers'] = int(num_layers_match.group(1)) if num_layers_match else 3
            params['hidden_size'] = int(hidden_size_match.group(1))
            params['seq_num'] = int(seq_num_match.group(1))
            return params
        except (AttributeError, ValueError):
            return None

    def _load_model(self, full_path: str, ModelClass: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], int]:
        """Loads a model from a given file path."""
        filename = os.path.basename(full_path)
        params = self._parse_model_name(filename)
        if params is None:
            logging.error(f"Could not parse parameters from model name: {filename}")
            return None, -1
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        checkpoint = torch.load(full_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        window_length = params['seq_num']
        
        model = ModelClass(num_features_joints=params['seq_num'], hidden_size=params['hidden_size'], num_layers=params['num_layers']).double()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        logging.info(f"Successfully loaded model: {filename}")
        return model, window_length