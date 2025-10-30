import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import logging
from collections import Counter
from torch.utils.data import DataLoader
import glob
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# --- NEW IMPORTS ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# -------------------
# Define custom two-tone colormap for the heatmap
custom_palette = [ '#67AB9F', '#EA6B66']    
sns.set_palette(custom_palette)
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_palette)

# --- 1. SETUP AND IMPORTS (Updated for new structure) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    import src
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from src.data import LoadSeqDataset
import src.models as model_zoo
# --- 2. HELPER FUNCTIONS ---

def majority_voting_last_n(model_out: pd.Series, n: int) -> pd.Series:
    """
    Applies a rolling majority vote to smooth out predictions.
    This helps to reduce noisy, high-frequency changes in the model's output.

    Args:
        model_out (pd.Series): The raw prediction series from the model.
        n (int): The size of the rolling window for the majority vote.

    Returns:
        pd.Series: The smoothed prediction series.
    """
    rolling_window = model_out.rolling(window=n, min_periods=1)
    smoothed_predictions = rolling_window.apply(lambda x: Counter(x).most_common(1)[0][0], raw=False).astype(int)
    return smoothed_predictions

def contact_detection_accuracy(df: pd.DataFrame, allowable_delay_ms: int = 50):
    """
    Calculates advanced detection metrics like TP, TN, FP, FN, and detection delays.
    This function evaluates performance in stable segments after an initial allowable delay.

    Args:
        df (pd.DataFrame): DataFrame containing timestamps, labels, and predictions.
        allowable_delay_ms (int): The time in milliseconds the model is allowed to detect a
                                  change in contact state without being penalized.

    Returns:
        tuple: A tuple containing TP, TN, FP, FN counts and lists of detection delays.
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    contact_delays, no_contact_delays = [], []
    df = df.reset_index(drop=True)
    # Identify points where the true label changes
    df['label_diff'] = df['label'].diff().fillna(0)
    change_events = df[df['label_diff'] != 0].index.tolist()
    if 0 not in change_events:
        change_events.insert(0, 0)
    if len(df) - 1 not in change_events:
        change_events.append(len(df) - 1)
        
    # Iterate through each stable segment (between label changes)
    for event_idx in range(len(change_events) - 1):
        idx = change_events[event_idx]
        end_of_segment_idx = change_events[event_idx + 1]
        if df.label[idx] == 1:
            start_time = df.time[idx]
            deadline = start_time + (allowable_delay_ms / 1000.0)
            first_detection_idx = -1
            for i in range(idx, end_of_segment_idx):
                if df.time[i] > deadline:
                    first_detection_idx = i
                    break
                if df.majority_voting[i] == 1:
                    delay = df.time[i] - start_time
                    contact_delays.append(delay)
                    first_detection_idx = i
                    break
            if first_detection_idx == -1:
                first_detection_idx = end_of_segment_idx
            stable_segment = df.iloc[first_detection_idx:end_of_segment_idx]
            TP += (stable_segment['majority_voting'] == 1).sum()
            FN += (stable_segment['majority_voting'] == 0).sum()
        else:
            start_time = df.time[idx]
            deadline = start_time + (allowable_delay_ms / 1000.0)
            first_detection_idx = -1
            for i in range(idx, end_of_segment_idx):
                if df.time[i] > deadline:
                    first_detection_idx = i
                    break
                if df.majority_voting[i] == 0:
                    delay = df.time[i] - start_time
                    no_contact_delays.append(delay)
                    first_detection_idx = i
                    break
            if first_detection_idx == -1:
                first_detection_idx = end_of_segment_idx
            stable_segment = df.iloc[first_detection_idx:end_of_segment_idx]
            TN += (stable_segment['majority_voting'] == 0).sum()
            FP += (stable_segment['majority_voting'] == 1).sum()
    return TP, TN, FP, FN, contact_delays, no_contact_delays


# --- 3. MAIN EVALUATION PIPELINE CLASS ---
class Evaluator:
    """
    Manages the end-to-end two-stage evaluation process.

    This class is responsible for:
    1. Loading the configuration.
    2. Instantiating and loading the pre-trained detection and localization models.
    3. Iterating through all test data files.
    4. Running the two-stage pipeline and reporting final performance metrics.
    5. ADDED: Generating a confusion matrix for the overall results.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Load both models upon initialization
        self.detection_model = self._load_model('detection_model')
        self.localization_model = self._load_model('localization_model')

    def _load_model(self, model_key: str) -> nn.Module:
        """
        Dynamically loads a model from the configuration.

        Args:
            model_key (str): The key for the model in the config file ('detection_model' or 'localization_model').

        Returns:
            nn.Module: The loaded and evaluated PyTorch model.
        """
        model_cfg = self.config[model_key]
        models_dir = self.config['project']['models_dir'].format(data_name=self.config['project']['data_name'])
        
        try:
            init_args = model_cfg['model_init_args']

            # --- Build the DMLClassificationNet from components ---
            if model_cfg['architecture'] == 'DMLClassificationNet':

                # 1. Build the Embedding Net
                embedding_net = model_zoo.Conv1DNet(
                    in_channels=init_args['in_channels'],
                    embedding_dim=init_args['embedding_dim'],
                    window_length=init_args['window_length'],
                    task=init_args['task']
                )

                # 2. Build the Classifier
                classifier = model_zoo.Classifier(
                    embedding_dim=init_args['embedding_dim'],
                    num_classes=init_args['num_classes']
                )

                # 3. Build the Wrapper
                model = model_zoo.DMLClassificationNet(embedding_net, classifier)

            # --- Fallback for your old models ---
            else:
                ModelClass = getattr(model_zoo, model_cfg['architecture'])
                model = ModelClass(**init_args)
        except Exception as e:
            raise TypeError(f"Error instantiating '{model_cfg['architecture']}'. Check 'model_init_args'. Error: {e}")
        
        weights_template = model_cfg['weights_file_template']
        format_context = {**model_cfg.get('model_init_args', {}), **model_cfg.get('filename_params', {})}
        final_weights_file = weights_template.format(**format_context)
        folder_name_ = model_cfg['type']
        folder_name = model_cfg['version']
        models_dir = f'{models_dir}/{folder_name_}/{folder_name}/{final_weights_file}'
        
        try:
            model.load_state_dict(torch.load(models_dir, map_location=self.device))
        except FileNotFoundError:
            logging.error(f"Model weights file not found at: {models_dir}")
            raise
        
        model.to(self.device).eval()
        logging.info(f"Successfully loaded weights for '{model_key}' from {models_dir}")
        return model

    def run(self):
        """
        Executes the main evaluation loop over all CSV files in the dataset directory.
        """
        data_directory = self.config['project']['dataset_dir'].format(data_name=self.config['project']['data_name'])
        all_csv_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)
        all_results_df = []
        total_stats = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'contact_delays': [], 'no_contact_delays': []}

        logging.info("Starting evaluation loop...")
        for file_path in all_csv_files:
            df_results, file_stats = self._process_file(file_path)
            if df_results is not None:
                all_results_df.append(df_results)
                for key in total_stats:
                    if isinstance(total_stats[key], list):
                        total_stats[key].extend(file_stats[key])
                    else:
                        total_stats[key] += file_stats[key]
        
        if not all_results_df:
            logging.warning("No data was processed. Exiting.")
            return

        final_df = pd.concat(all_results_df, ignore_index=True)
        self._report_results(final_df, total_stats)

    def _process_file(self, file_path: str) -> tuple:
        """
        Processes a single data file through the two-stage pipeline.

        This method performs the following steps:
        1. Runs the high-performance detection model on the entire file.
        2. Applies a rolling majority-voting filter to smooth the detection predictions.
        3. Identifies the specific time steps where contact was detected.
        4. Runs the more complex localization model ONLY on these identified time steps.
        5. Combines the results and calculates performance statistics for the file.

        Args:
            file_path (str): The path to the CSV file to be processed.
        
        Returns:
            tuple: A DataFrame with detailed results and a dictionary of performance
                   stats for the file, or (None, None) if the file is too short.
        """
        # --- 1. Load Data and Configuration ---
        full_data_df = pd.read_csv(file_path)
        
        det_model_cfg = self.config['detection_model']
        loc_model_cfg = self.config['localization_model']
        
        label_val = next((num for name, num in self.config['labels'].items() if name in file_path), 0)
        label_to_pass = torch.tensor(label_val)
        selected_features = [f'e{i}' for i in range(self.config['project']['dof'])]
        
        # --- Stage 1: Contact Detection ---
        det_params = det_model_cfg['evaluator_params']
        detection_dataset = LoadSeqDataset(
            file_path=file_path, label=label_to_pass, selected_features=selected_features,
            window_length=det_params['window_length'], gap=det_params['gap'],
            mode='val', data_df=full_data_df
        )
        if len(detection_dataset) == 0:
            return None, None

        detection_loader = DataLoader(detection_dataset, batch_size=det_params['batch_size'], shuffle=False)
        detection_predictions = []
        with torch.no_grad():
            for inputs, _ in detection_loader:
                outputs = self.detection_model.prediction(inputs.to(self.device))
                detection_predictions.extend(outputs.cpu().numpy().flatten())
        
        # --- Process Detection Results ---
        df_results = pd.DataFrame({
            'time': detection_dataset.times,
            'label_loc': detection_dataset.labels,
            'detection_pred': detection_predictions
        })
        df_results['detection_pred_smooth'] = majority_voting_last_n(
            df_results['detection_pred'], det_params['n_majority_voting']
        )
        df_results['pipeline_final_pred'] = 0
        
        contact_indices_det = df_results[df_results['detection_pred_smooth'] == 1].index.to_numpy()

        # --- Stage 2: Contact Localization ---
        if contact_indices_det.size > 0:
            loc_params = loc_model_cfg['evaluator_params']
            localization_dataset = LoadSeqDataset(
                file_path=file_path, label=label_to_pass, selected_features=selected_features,
                window_length=loc_params['window_length'], gap=loc_params['gap'],
                mode='val', data_df=full_data_df
            )
            
            index_diff = (loc_params['window_length'] - det_params['window_length']) // det_params['gap']
            contact_indices_loc = contact_indices_det - index_diff
            
            valid_mask = (contact_indices_loc >= 0) & (contact_indices_loc < len(localization_dataset))
            valid_loc_indices = contact_indices_loc[valid_mask]
            original_indices_to_update = contact_indices_det[valid_mask]

            if valid_loc_indices.size > 0:
                sequences_for_loc = torch.stack([localization_dataset[i][0] for i in valid_loc_indices]).to(self.device)
                with torch.no_grad():
                    localization_preds = self.localization_model.prediction(sequences_for_loc).cpu().numpy()
                df_results.loc[original_indices_to_update, 'pipeline_final_pred'] = localization_preds

        # --- Calculate Performance Metrics ---
        detection_metric_df = df_results.copy()
        detection_metric_df['label'] = (detection_metric_df['label_loc'] > 0).astype(int)
        detection_metric_df['majority_voting'] = detection_metric_df['detection_pred_smooth']
        
        TP, TN, FP, FN, c_delays, nc_delays = contact_detection_accuracy(
            detection_metric_df,
            allowable_delay_ms=det_params['allowable_delay_ms']
        )
        file_stats = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'contact_delays': c_delays, 'no_contact_delays': nc_delays}
        
        return df_results, file_stats

# --- METHOD TO BE UPDATED (Corrected Signature) ---
    def _plot_confusion_matrix(self, y_true, y_pred, class_names, class_labels, task_type: str): # Added task_type here
        """Generates and saves a confusion matrix plot based on task type."""
        try:
            # Generate the matrix using the explicit label numbers
            # Ensure labels parameter includes all unique values present in y_true or y_pred
            # that we want to plot, even if some aren't in class_labels (though they should be)
            present_labels = sorted(list(set(y_true) | set(y_pred)))
            plot_labels = sorted(list(set(class_labels) | set(present_labels))) # Combine expected and actual labels

            cm = confusion_matrix(y_true, y_pred, labels=plot_labels) # Use combined labels
            logging.info(cm)
            # Adjust class_names if plot_labels has extra unexpected labels
            plot_class_names = []
            label_to_name = dict(zip(class_labels, class_names))
            for label in plot_labels:
                 plot_class_names.append(label_to_name.get(label, f"Unknown ({label})"))

            # --- Font Size Adjustments ---
            label_fontsize = 10
            tick_fontsize = 10
            annotation_fontsize = 9 # Font size for numbers inside the heatmap
            # ---------------------------

            plt.figure(figsize=(max(7, len(plot_class_names)*0.8), max(5, len(plot_class_names)*0.6))) # Adjust figure size slightly if needed

            sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap,
                        xticklabels=plot_class_names, yticklabels=plot_class_names,
                        annot_kws={"size": annotation_fontsize}) # Set annotation font size

            # Set font sizes for title and labels
            plt.ylabel('True Label', fontsize=label_fontsize)
            plt.xlabel('Predicted Label', fontsize=label_fontsize)

            # Set font sizes for tick labels (optional, but good for consistency)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize, rotation=0) # Keep y-axis labels horizontal

            plt.tight_layout()

            # Determine save path (same logic as before)
            save_dir_base = 'models' # Default
            data_name = self.config.get('project', {}).get('data_name', 'results')
            try:
                 project_root = Path(__file__).resolve().parents[2]
                 save_dir_base = os.path.join(project_root, 'models', data_name)
                 if not os.path.isdir(save_dir_base):
                     save_dir_base = os.path.join(project_root, 'models') # fallback
            except Exception:
                 project_root = '.' # fallback if structure is unexpected
                 save_dir_base = os.path.join(project_root,'models', data_name)


            os.makedirs(save_dir_base, exist_ok=True)
            # Use task_type in the filename
            save_path = os.path.join(save_dir_base, f'confusion_matrix_{task_type}_{data_name}.pdf')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"{task_type.replace('_', ' ').capitalize()} confusion matrix saved to: {save_path}")
            plt.close()

        except Exception as e:
            logging.error(f"Failed to generate {task_type} confusion matrix plot: {e}", exc_info=True)
    
    def _report_results(self, final_df: pd.DataFrame, total_stats: dict):
        """
        Logs final metrics and generates separate confusion matrices.
        Localization matrix now excludes 'No Contact' class entirely.
        """
        # --- Stage 1 Report (Identical) ---
        logging.info("\n--- Stage 1: Detection Model Performance ---")
        total_all = total_stats['TP'] + total_stats['TN'] + total_stats['FP'] + total_stats['FN']
        ModelAccuracy = (total_stats['TP'] + total_stats['TN']) / total_all * 100 if total_all > 0 else 0
        DetectionFailureRate = total_stats['FN'] / (total_stats['TP'] + total_stats['FN']) * 100 if (total_stats['TP'] + total_stats['FN']) > 0 else 0
        FalseAlarmRate = total_stats['FP'] / (total_stats['TN'] + total_stats['FP']) * 100 if (total_stats['TN'] + total_stats['FP']) > 0 else 0
        avg_contact_delay = np.nanmean(total_stats['contact_delays']) if total_stats['contact_delays'] else 0
        avg_no_contact_delay = np.nanmean(total_stats['no_contact_delays']) if total_stats['no_contact_delays'] else 0

        logging.info(f"Detection Accuracy: {ModelAccuracy:.2f}%")
        logging.info(f"Detection Failure Rate: {DetectionFailureRate:.2f}% (Missed Detections)")
        logging.info(f"False Alarm Rate: {FalseAlarmRate:.2f}%")
        logging.info(f"Average Contact Detection Delay: {avg_contact_delay:.4f} seconds")
        logging.info(f"Average No-Contact Detection Delay: {avg_no_contact_delay:.4f} seconds")

        # --- Stage 2 Report (Identical) ---
        logging.info("\n--- Stage 2: Localization Model Performance ---")
        # Filter for instances where true contact occurred AND prediction indicates contact
        true_positive_detections_df = final_df[(final_df['label_loc'] > 0) & (final_df['pipeline_final_pred'] > 0)]
        if not true_positive_detections_df.empty:
            correctly_localized_tps = (true_positive_detections_df['pipeline_final_pred'] == true_positive_detections_df['label_loc']).sum()
            loc_accuracy_on_tps = (correctly_localized_tps / len(true_positive_detections_df)) * 100
            logging.info(f"Localization Accuracy on Correctly Detected Contacts: {loc_accuracy_on_tps:.2f}%")
        else:
            # Check if there were true contacts at all, even if none were detected
             if (final_df['label_loc'] > 0).any():
                  logging.warning("No contacts were correctly *detected* by the pipeline to calculate localization accuracy.")
             else:
                  logging.info("No true contact samples present in the dataset.")


        # --- Generate Detection Confusion Matrix (Identical) ---
        logging.info("\n--- Generating Detection Confusion Matrix ---")
        try:
            y_true_detect = (final_df['label_loc'] > 0).astype(int)
            y_pred_detect = final_df['detection_pred_smooth'].astype(int)
            detect_class_names = ['No Contact', 'Contact']
            detect_labels = [0, 1]
            self._plot_confusion_matrix(y_true_detect, y_pred_detect, detect_class_names, detect_labels, task_type='detection')
        except Exception as e:
            logging.error(f"Error during detection confusion matrix generation: {e}", exc_info=True)

        # --- Generate Localization Confusion Matrix (MODIFIED: Excludes No Contact) ---
        logging.info("\n--- Generating Localization Confusion Matrix (Contact Classes Only) ---")
        try:
            # Filter the dataframe for rows where BOTH true label AND prediction indicate contact (>0)
            # This completely removes the 'No Contact' class from consideration for this matrix
            contact_only_df = final_df[(final_df['label_loc'] > 0) & (final_df['pipeline_final_pred'] > 0)].copy()

            if not contact_only_df.empty:
                y_true_loc = contact_only_df['label_loc'].astype(int)
                y_pred_loc = contact_only_df['pipeline_final_pred'].astype(int)

                # Get class names/labels excluding 'no_contact' (label 0)
                loc_labels_sorted = sorted([item for item in self.config.get('labels', {}).items() if item[1] > 0], key=lambda item: item[1])
                loc_class_names = [name for name, label_num in loc_labels_sorted]
                loc_label_nums = [label_num for name, label_num in loc_labels_sorted] # These are the labels > 0

                if not loc_class_names:
                    logging.warning("No contact labels (>0) found in config 'labels'. Skipping localization confusion matrix.")
                else:
                    # Pass ONLY the contact class names and labels
                    self._plot_confusion_matrix(y_true_loc, y_pred_loc, loc_class_names, loc_label_nums, task_type='localization_contact_only') # Changed task_type for filename
            else:
                 logging.warning("No samples where both true label and prediction indicated contact (>0). Cannot generate contact-only localization confusion matrix.")

        except KeyError as e:
            logging.error(f"Config missing key {e}. Skipping localization confusion matrix.", exc_info=True)
        except Exception as e:
            logging.error(f"Error during localization confusion matrix preparation: {e}", exc_info=True)    