import numpy as np
import os
import re
import torch
import sys
import logging
from typing import Tuple, Optional, Dict, List
from collections import Counter
import time

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Model Imports ---
try:
    from pipelines_transferLearning.src import models as model_zoo

except ImportError as e:
    logging.error(f"Fatal: Could not import model definitions from 'src.models'. Error: {e}")
    sys.exit(1)


class ContactAI:
    def __init__(self, ai_model_config: dict, num_features: int):
        """
        Initializes the ContactAI models based on a configuration dictionary.
        
        Args:
            ai_model_config (dict): The dictionary loaded from the AI model's YAML file.
            num_features (int): The number of features from the robot (DoF).
        """
        logging.info("--- 🧠 Initializing Contact AI ---")
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        self.ai_config = ai_model_config
        self.project_config = ai_model_config.get('project', {})

        # --- 1. Load Detection Model ---
        det_model_cfg = self.ai_config.get('detection_model')
        if not det_model_cfg:
            logging.error("AI config is missing 'detection_model' section. Exiting.")
            sys.exit(1)
        
        logging.info("Loading contact detection model...")
        self.detection_model, self.window_length, self.detection_model_arch = self._load_model_from_config(
            det_model_cfg
        )
        
        # --- 2. Load Localization Model ---
        loc_model_cfg = self.ai_config.get('localization_model')
        if not loc_model_cfg:
            logging.error("AI config is missing 'localization_model' section. Exiting.")
            sys.exit(1)
            
        logging.info("Loading contact localization model...")
        self.localization_model, self.loc_window_length, self.localization_model_arch = self._load_model_from_config(
            loc_model_cfg
        )

        # --- 3. Final Checks and Setup ---
        if self.detection_model is None or self.localization_model is None:
            logging.error("One or both models failed to load. Please check your config paths.")
            sys.exit(1)

        self.majority_voting_window = det_model_cfg.get('evaluator_params', {}).get('n_majority_voting', 14)
        self.detection_history: List[int] = []
        
        if self.window_length != self.loc_window_length:
            logging.warning(f"Window lengths for detection ({self.window_length}) and localization ({self.loc_window_length}) models differ.")
        
        self.master_window_length = max(self.window_length, self.loc_window_length)
        self.window = np.zeros((self.num_features, self.master_window_length), dtype=np.float64)
        logging.info(f"ContactAI initialized. Master window size: {self.master_window_length}")


    def _build_model_path(self, model_cfg: dict) -> str:
        """
        Constructs the full path to the model weights file from the config.
        """
        models_dir_template = self.project_config.get('models_dir', '')
        data_name = self.project_config.get('data_name', '')
        
        try:
            models_dir = models_dir_template.format(data_name=data_name)
        except KeyError:
            models_dir = models_dir_template
        
        weights_template = model_cfg.get('weights_file_template')
        model_type = model_cfg.get('type', '')
        model_version = str(model_cfg.get('version', '')) # Ensure version is a string
        
        format_context = {
            **model_cfg.get('model_init_args', {}),
            **model_cfg.get('filename_params', {})
        }

        if not weights_template:
            logging.error(f"Config for architecture '{model_cfg.get('architecture')}' is missing 'weights_file_template'.")
            return None
            
        try:
            final_weights_file = weights_template.format(**format_context)
        except KeyError as e:
            # Handle the 'acc' vs 'accuracy' mismatch automatically
            if 'acc' in str(e) and 'acc' not in format_context and 'accuracy' in format_context:
                logging.warning(f"Formatting 'weights_file_template' failed. Trying to map 'accuracy' to 'acc'.")
                format_context['acc'] = format_context['accuracy']
                try:
                    final_weights_file = weights_template.format(**format_context)
                except KeyError as e2:
                     logging.error(f"Failed to format weights filename (after 'acc' fix). Missing key: {e2}")
                     return None
            else:
                logging.error(f"Failed to format weights filename. Missing key: {e}")
                logging.error(f"Available keys for formatting: {list(format_context.keys())}")
                return None

        full_path = os.path.join(project_root, models_dir, model_type, model_version, final_weights_file)
        return full_path


    def _load_model_from_config(self, model_cfg: dict) -> Tuple[Optional[torch.nn.Module], int, str]:
        """
        Dynamically instantiates and loads a model from the config dictionary.
        """
        arch_name = model_cfg.get('architecture')
        init_args = model_cfg.get('model_init_args', {})

        if not arch_name:
            logging.error("Model config is missing 'architecture' key.")
            return None, 0, ""

        model = None
        try:
            # --- Instantiate the Model ---
            if arch_name == 'DMLClassificationNet':
                # --- THIS IS THE FIX ---
                # We separate the arguments for each sub-model.
                
                # 1. Build arguments for Conv1DNet
                conv1d_args = {
                    'in_channels': init_args.get('in_channels'),
                    'embedding_dim': init_args.get('embedding_dim'),
                    'window_length': init_args.get('window_length'),
                    'task': init_args.get('task')
                }
                # Filter out None values in case a key is missing
                conv1d_args = {k: v for k, v in conv1d_args.items() if v is not None}

                EmbeddingModelClass = getattr(model_zoo, 'Conv1DNet')
                embedding_net = EmbeddingModelClass(**conv1d_args)
                
                # 2. Build arguments for Classifier
                classifier_args = {
                    'embedding_dim': embedding_net.embedding_dim,
                    'num_classes': init_args.get('num_classes')
                }
                classifier_args = {k: v for k, v in classifier_args.items() if v is not None}
                
                classifier = model_zoo.Classifier(**classifier_args)
                
                # 3. Build the wrapper. It ONLY takes the two modules as args.
                model = model_zoo.DMLClassificationNet(embedding_net, classifier)
                # --- END FIX ---
            else:
                # Standard loading for cnnBiLSTM, TransformerModel
                ModelClass = getattr(model_zoo, arch_name)
                model = ModelClass(**init_args)
        
        except AttributeError:
            logging.error(f"Model architecture '{arch_name}' not found in 'src.models'. Check for typos.")
            return None, 0, ""
        except Exception as e:
            logging.error(f"Error instantiating '{arch_name}' with args {init_args}.")
            logging.error(f"Original Error: {e}", exc_info=True)
            return None, 0, ""

        # --- Build the Path to the Weights ---
        full_path = self._build_model_path(model_cfg)
        
        if not full_path:
             logging.error(f"Could not construct model path from config: {model_cfg}")
             return None, 0, ""
             
        if not os.path.exists(full_path):
            # Try removing project_root (in case models_dir is an absolute path or relative to a different root)
            alt_path = full_path.replace(os.path.join(project_root, ""), "")
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                logging.error(f"Model weights file not found at: {full_path}")
                logging.error(f"(Also checked: {alt_path})")
                return None, 0, ""

        # --- Load the Weights ---
        try:
            checkpoint = torch.load(full_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            logging.info(f"✅ Successfully loaded model: {os.path.basename(full_path)}")
            
            window_length_key = 'num_features' if 'num_features' in init_args else 'window_length'
            window_length = init_args.get(window_length_key, -1)
            
            if window_length == -1:
                logging.error(f"Could not find '{window_length_key}' in model_init_args.")
                return None, 0, ""
                
            return model, window_length, arch_name
        
        except Exception as e:
            logging.error(f"Failed to load model weights from {full_path}: {e}", exc_info=True)
            return None, 0, ""


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
        
        features_dict = robot_data.get("features")
        if not features_dict:
            return 0, 0, -1
        # Convert the dictionary of features into a list of values
        feature_values = list(features_dict.values())
        # ---------------------
        

        new_column = np.array(feature_values, dtype=np.float64).reshape((self.num_features, 1))
        self.window = np.append(self.window[:, 1:], new_column, axis=1)

        with torch.no_grad():
            detection_window = self.window[:, -self.window_length:]
            detection_input = torch.from_numpy(detection_window).unsqueeze(0).float().to(self.device)
            raw_detection_pred = self.detection_model.prediction(detection_input).item()

            self.detection_history.append(raw_detection_pred)
            smoothed_detection_pred = raw_detection_pred
            if len(self.detection_history) >= self.majority_voting_window:
                smoothed_detection_pred = self._majority_vote()

            localization_pred = -1
            if smoothed_detection_pred == 1:
                localization_window = self.window[:, -self.loc_window_length:]
                localization_input = torch.from_numpy(localization_window).unsqueeze(0).float().to(self.device)
                localization_pred = self.localization_model.prediction(localization_input).item()
                
        return raw_detection_pred, smoothed_detection_pred, localization_pred