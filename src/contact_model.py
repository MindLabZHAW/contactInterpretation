import numpy as np
import os
import re
import torch
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dynamically add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logging.info(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your custom model architectures
try:
    from pipelines.models.cnnLSTM_contactDetection import cnnLSTM as ContactDetectionModel
    from pipelines.models.cnnLSTM_contactLocalization import cnnLSTM as ContactLocalizationModel
except ImportError:
    logging.error("Could not import model definitions from 'pipelines' directory. Ensure the path is correct and models are available.")
    sys.exit(1)

class ContactDetectorAI:
    """
    Manages the AI model, including dynamic loading from the console, state management
    (sliding window), and real-time inference.
    """
    def __init__(self, robot_dof=7):
        self.robot_dof = robot_dof
        self.model, self.window_length = self._select_and_load_model()
        
        if self.model is None:
            logging.error("No model was loaded. The system cannot perform contact detection.")
            self.window = np.zeros([self.robot_dof, 1])
        else:
            self.window = np.zeros([self.robot_dof, self.window_length])

    def predict_contact(self, robot_data: dict) -> bool:
        if self.model is None:
            return False

        e_q = robot_data.get("e_q")
        if e_q is None:
            logging.warning("Field 'e_q' not found in robot data, cannot run inference.")
            return False

        new_column = np.array(e_q).reshape((self.robot_dof, 1))
        self.window = np.append(self.window[:, 1:], new_column, axis=1)
        
        data_input = torch.tensor([self.window], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            contact_result = self.model.prediction(data_input).item()

        return contact_result == 1

    def _parse_model_name(self, filename):
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

    def _select_and_load_model(self):
        base_models_dir = os.path.join(project_root, 'pipelines', 'trained_models')
        if not os.path.isdir(base_models_dir):
            logging.error(f"Base model directory not found at: {base_models_dir}")
            return None, -1

        try:
            subdirectories = [d for d in os.listdir(base_models_dir) if os.path.isdir(os.path.join(base_models_dir, d))]
            if not subdirectories: return None, -1
            
            print("\n--- Please select a model directory ---")
            for i, dirname in enumerate(subdirectories): print(f"[{i}] {dirname}")
            dir_choice = int(input("Select by number: "))
            selected_dir_name = subdirectories[dir_choice]
            search_dir = os.path.join(base_models_dir, selected_dir_name)
        except (ValueError, IndexError):
            logging.error("Invalid directory selection.")
            return None, -1

        available_models = [
            (os.path.relpath(os.path.join(root, file), search_dir), os.path.join(root, file))
            for root, _, files in os.walk(search_dir) for file in files if file.endswith('.pth')
        ]
        if not available_models: return None, -1

        print(f"\n--- Available Models in '{selected_dir_name}' ---")
        for i, (display_path, _) in enumerate(available_models): print(f"[{i}] {display_path}")
        
        try:
            model_choice = int(input("Select by number: "))
            display_path, full_path = available_models[model_choice]
            filename = os.path.basename(full_path)
        except (ValueError, IndexError):
            logging.error("Invalid model selection.")
            return None, -1
        
        params = self._parse_model_name(filename)
        if params is None: return None, -1
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(full_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        window_length = params['seq_num']
        model_type = 'contactDetection' if 'contactDetection' in display_path else 'contactLocalization'
        
        logging.info(f"Loading model '{filename}' with params: {params}")
        ModelClass = ContactDetectionModel if model_type == 'contactDetection' else ContactLocalizationModel
        model = ModelClass(num_features_joints=window_length, hidden_size=params['hidden_size'], num_layers=params['num_layers'])
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self.device = device
        
        logging.info(f"Successfully loaded model: {filename}")
        return model, window_length

