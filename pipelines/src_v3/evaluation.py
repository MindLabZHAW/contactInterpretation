import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import logging
import glob
import yaml
import argparse
from collections import Counter
from torch.utils.data import DataLoader

# --- 1. SETUP AND IMPORTS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
project_root = os.getcwd().replace('pipelines', '')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from pipelines.src_v3.dataset_loader import LoadSeqDataset
import pipelines.models.all_models as model_zoo

# --- 2. HELPER FUNCTIONS ---
# (These functions are unchanged)
def majority_voting_last_n(model_out, n):
    model_out_series = pd.Series(model_out)
    rolling_window = model_out_series.rolling(window=n, min_periods=1)
    smoothed_predictions = rolling_window.apply(lambda x: Counter(x).most_common(1)[0][0], raw=False).astype(int)
    return smoothed_predictions

def contact_detection_accuracy(df, allowable_delay_ms=50):
    TP, TN, FP, FN = 0, 0, 0, 0
    contact_delays, no_contact_delays = [], []
    df = df.reset_index(drop=True)
    df['label_diff'] = df['label'].diff().fillna(0)
    change_events = df[df['label_diff'] != 0].index.tolist()
    if 0 not in change_events:
        change_events.insert(0, 0)
    if len(df) - 1 not in change_events:
        change_events.append(len(df) - 1)
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
class EvaluationPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.detection_model = self._load_model('detection')
        self.localization_model = self._load_model('localization')

    def _load_model(self, model_key):
        """Loads a model using init args and a filename template from the config."""
        model_config = self.config[f'{model_key}_model']
        model_dir = os.path.join(
            project_root,
            self.config['paths']['models_dir'].format(data_name=self.config['data_name']),
            model_config['type'],
            model_config['version']
        )
        
        # 1. Get model initialization arguments
        init_args = model_config.get('model_init_args')
        if not init_args:
            raise ValueError(f"'model_init_args' not found for '{model_key}' in config.")

        # 2. Instantiate the model
        architecture_name = model_config.get('architecture')
        try:
            ModelClass = getattr(model_zoo, architecture_name)
            model = ModelClass(**init_args)
        except Exception as e:
            raise TypeError(f"Error instantiating '{architecture_name}'. Check 'model_init_args'. Error: {e}")

        # 3. Build the weights filename from the template
        weights_template = model_config.get('weights_file_template')
        if not weights_template:
            raise ValueError(f"'weights_file_template' not found for '{model_key}' in config.")
        
        # Create a context dictionary for formatting the filename string
        filename_params = model_config.get('filename_params', {})
        # Combine init args and filename-specific params. Filename params will overwrite if keys are the same.
        format_context = {**init_args, **filename_params}
        
        final_weights_file = weights_template.format(**format_context)
        model_path = os.path.join(model_dir, final_weights_file)
        
        # 4. Load the weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            logging.error(f"Model weights file not found at: {model_path}")
            raise

        model.to(self.device).eval()
        logging.info(f"Successfully loaded weights for '{model_key}' from {model_path}")
        
        return model
    
    def run(self):
        data_directory = os.path.join(project_root, self.config['paths']['dataset_dir'].format(data_name=self.config['data_name']))
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

    def _process_file(self, file_path):
        cfg = self.config
        full_data_df = pd.read_csv(file_path)
        label_val = next((num for name, num in cfg['labels'].items() if name in file_path), 0)
        label_to_pass = torch.tensor(label_val)
        selected_features = [f'e{i}' for i in range(cfg['dof'])]
        
        det_loader_params = cfg['detection_model']['data_loader_params']
        detection_dataset = LoadSeqDataset(
            file_path=file_path, label=label_to_pass, selected_features=selected_features,
            seq_num=int(det_loader_params['seq_num']), gap=int(det_loader_params['gap']),
            mode='val', data_df=full_data_df
        )
        if len(detection_dataset) == 0: return None, None

        detection_loader = DataLoader(detection_dataset, batch_size=cfg['evaluation_params']['batch_size'], shuffle=False)
        detection_predictions = []
        with torch.no_grad():
            for inputs, _ in detection_loader:
                outputs = self.detection_model.prediction(inputs.to(self.device))
                detection_predictions.extend(outputs.cpu().numpy().flatten())
        
        df_results = pd.DataFrame({
            'time': detection_dataset.times, 'label_loc': detection_dataset.labels, 'detection_pred': detection_predictions
        })
        df_results['detection_pred_smooth'] = majority_voting_last_n(df_results['detection_pred'], cfg['evaluation_params']['n_majority_voting'])
        df_results['pipeline_final_pred'] = 0
        contact_indices_det = df_results[df_results['detection_pred_smooth'] == 1].index.to_numpy()

        if contact_indices_det.size > 0:
            loc_loader_params = cfg['localization_model']['data_loader_params']
            localization_dataset = LoadSeqDataset(
                file_path=file_path, label=label_to_pass, selected_features=selected_features,
                seq_num=int(loc_loader_params['seq_num']), gap=int(loc_loader_params['gap']),
                mode='val', data_df=full_data_df
            )
            index_diff = (int(loc_loader_params['seq_num']) - int(det_loader_params['seq_num'])) // int(det_loader_params['gap'])
            contact_indices_loc = contact_indices_det - index_diff
            valid_mask = (contact_indices_loc >= 0) & (contact_indices_loc < len(localization_dataset))
            valid_loc_indices = contact_indices_loc[valid_mask]
            original_indices_to_update = contact_indices_det[valid_mask]

            if valid_loc_indices.size > 0:
                sequences_for_loc = [localization_dataset[i][0] for i in valid_loc_indices]
                loc_batch = torch.stack(sequences_for_loc).to(self.device)
                with torch.no_grad():
                    localization_preds = self.localization_model.prediction(loc_batch).cpu().numpy()
                df_results.loc[original_indices_to_update, 'pipeline_final_pred'] = localization_preds

        detection_metric_df = df_results.copy()
        detection_metric_df['label'] = (detection_metric_df['label_loc'] > 0).astype(int)
        detection_metric_df['majority_voting'] = detection_metric_df['detection_pred_smooth']
        TP, TN, FP, FN, c_delays, nc_delays = contact_detection_accuracy(
            detection_metric_df, 
            allowable_delay_ms=cfg['evaluation_params']['allowable_delay_ms']
        )
        file_stats = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'contact_delays': c_delays, 'no_contact_delays': nc_delays}
        return df_results, file_stats

    def _report_results(self, final_df, total_stats):
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
        logging.info("\n--- Stage 2: Localization Model Performance ---")
        true_positive_detections_df = final_df[(final_df['label_loc'] > 0) & (final_df['pipeline_final_pred'] > 0)]
        if not true_positive_detections_df.empty:
            correctly_localized_tps = (true_positive_detections_df['pipeline_final_pred'] == true_positive_detections_df['label_loc']).sum()
            loc_accuracy_on_tps = (correctly_localized_tps / len(true_positive_detections_df)) * 100
            logging.info(f"Localization Accuracy on Correctly Detected Contacts: {loc_accuracy_on_tps:.2f}%")
        else:
            logging.warning("No correct contact detections were made to calculate localization accuracy.")


# --- 4. SCRIPT ENTRY POINT ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the two-stage contact detection and localization pipeline.")
    parser.add_argument('--config', type=str, default='pipelines/config/config_franka_main.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {args.config}")
        sys.exit(1)
    
    pipeline = EvaluationPipeline(config)
    pipeline.run()