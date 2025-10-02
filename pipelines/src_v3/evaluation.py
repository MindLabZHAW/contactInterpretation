import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import logging
import time
import glob
from collections import Counter
from torch.utils.data import DataLoader, Subset

# --- Add project root to path ---
project_root = os.getcwd().replace('pipelines', '')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipelines.src_v3.dataset_loader import LoadSeqDataset
from pipelines.models.cnnLSTM_contactDetection import cnnLSTM as ContactDetectionModel
from pipelines.models.cnnLSTM_contactLocalization import cnnLSTM as ContactLocalizationModel

# --- 1. Helper Functions ---
def majority_voting_last_n(model_out, n):
    smoothed_predictions = model_out.copy()
    for i in range(n, len(model_out)):
        window = model_out.iloc[i-n:i]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed_predictions.iloc[i-1] = most_common
    return smoothed_predictions

def contact_detection_accuracy(df, allowable_delay=30):
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
        
        segment = df.iloc[idx:change_events[event_idx+1]]
        is_contact_segment = df.label[idx] == 1

        if is_contact_segment:
            TP += (segment['majority_voting'] == 1).sum()
            FN += (segment['majority_voting'] == 0).sum()
            
            first_detection = segment[segment['majority_voting'] == 1]
            if not first_detection.empty:
                delay = first_detection.iloc[0]['time'] - segment.iloc[0]['time']
                if delay < (allowable_delay * (df.time.iloc[1] - df.time.iloc[0])):
                        contact_delays.append(delay)
        else: # No-contact segment
            TN += (segment['majority_voting'] == 0).sum()
            FP += (segment['majority_voting'] == 1).sum()

    return TP, TN, FP, FN, contact_delays, no_contact_delays

# --- 2. Main Evaluation Functions ---
def find_best_model_from_csv(results_csv_path):
    """Finds the hyperparameters of the best model from the results CSV."""
    if not os.path.exists(results_csv_path):
        logging.error(f"Results CSV not found at: {results_csv_path}")
        return None
    df = pd.read_csv(results_csv_path)
    best_model_row = df.loc[df['ACC'].idxmax()]
    logging.info(f"Found best model with accuracy {best_model_row['ACC']:.2f}%")
    return best_model_row.to_dict()

def evaluate_best_model(task_type):
    """
    Main function to orchestrate the evaluation of the best model for a given task.
    """
    data_name = 'franka_main'
    dof = 7
    batch_size = 1024
    n_majority_voting = 14
    
    if task_type == 'contact_detection':
        model_class = ContactDetectionModel
        results_csv_path = os.path.join(project_root, 'pipelines', 'trained_models', data_name, 'contact_detection_v3','64', 'hyperparameter_results.csv')
    elif task_type == 'contact_localization':
        model_class = ContactLocalizationModel
        results_csv_path = os.path.join(project_root, 'pipelines', 'trained_models', data_name, 'contact_localization_v3','67', 'hyperparameter_results_localization.csv')
    else:
        logging.error("Invalid task type specified.")
        return

    best_hyperparams = find_best_model_from_csv(results_csv_path)
    if not best_hyperparams:
        return

    model_dir = os.path.dirname(results_csv_path)
    model_name = f"numLayer{int(best_hyperparams['num_layer'])}_hiddenSize{int(best_hyperparams['hidden_size'])}_seq_num{int(best_hyperparams['seq_num'])}_gap{int(best_hyperparams['gap'])}_accuracy{best_hyperparams['ACC']:.2f}.pth"
    model_path = os.path.join(model_dir, model_name)

    model = model_class(num_features_joints=int(best_hyperparams['seq_num']), hidden_size=int(best_hyperparams['hidden_size']), num_layers=int(best_hyperparams['num_layer']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    data_directory = os.path.join(project_root, 'dataset', data_name, 'labeled_data')
    dict_label = {'link7': 7, 'link6': 6, 'link5': 5, 'link4': 4, 'link3': 3, 'link2': 2, 'link1': 1, 'no_contact': 0}
    selected_features = [f'e{i}' for i in range(dof)]
    all_csv_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)

    total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0
    all_contact_delays, all_no_contact_delays = [], []
    
    # Initialize a list to store results for each file
    per_file_results = []

    for file_path in all_csv_files:
        label_val = 0
        for name, number in dict_label.items():
            if name in file_path:
                label_val = number
                break
        
        trial_dataset = LoadSeqDataset(file_path, torch.tensor(label_val), selected_features, seq_num=int(best_hyperparams['seq_num']), gap=1, mode='val')
                
        test_loader = DataLoader(trial_dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                if task_type == 'contact_detection':
                    outputs = (torch.sigmoid(model(inputs)) > 0.5).int()
                else:
                    outputs = torch.argmax(model(inputs), dim=1) + 1
                predictions.extend(outputs.cpu().numpy())

        df_results = pd.DataFrame({
            'time': trial_dataset.times,
            'label': trial_dataset.labels,
            'model_out': predictions
        })
        
        if task_type == 'contact_detection':
            df_results['label'] = (df_results['label'] > 0).astype(int)

        df_results['majority_voting'] = majority_voting_last_n(df_results['model_out'], n_majority_voting)

        TP, TN, FP, FN, contact_delays, _ = contact_detection_accuracy(df_results)

        # Calculate and store the accuracy for the current file
        file_total = TP + TN + FP + FN
        if file_total > 0:
            file_accuracy = (TP + TN) / file_total * 100
            file_name = os.path.basename(file_path)
            per_file_results.append({'filename': file_name, 'accuracy': file_accuracy})
        
        # Aggregate totals for overall results
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN
        all_contact_delays.extend(contact_delays)

    # Display the per-file results after the loop
    logging.info("\n--- Accuracy Per File ---")
    results_df = pd.DataFrame(per_file_results)
    logging.info(f"\n{results_df.to_string()}")


    # Display the overall results
    logging.info("\n--- Overall Evaluation Results ---")
    ModelAccuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN) * 100 if (total_TP + total_TN + total_FP + total_FN) > 0 else 0
    DetectionFailureRate = total_FN / (total_TP + total_FN) * 100 if (total_TP + total_FN) > 0 else 0
    FalseAlarmRate = total_FP / (total_TN + total_FP) * 100 if (total_TN + total_FP) > 0 else 0
    avg_contact_delay = np.nanmean(all_contact_delays) if all_contact_delays else 0

    logging.info(f"Overall Model Accuracy: {ModelAccuracy:.2f}%")
    logging.info(f"Detection Failure Rate: {DetectionFailureRate:.2f}%")
    logging.info(f"False Alarm Rate: {FalseAlarmRate:.2f}%")
    logging.info(f"Average Contact Detection Delay: {avg_contact_delay:.4f} seconds")
    # Display the per-file results after the loop
    logging.info("\n--- Accuracy Per File ---")
    results_df = pd.DataFrame(per_file_results)
    logging.info(f"\n{results_df.to_string()}")

    # --- ADD THIS SECTION TO SAVE THE FILE ---
    # Define the output path for the CSV file
    output_path = os.path.join(project_root, 'per_file_accuracy_results.csv')
    
    # Save the DataFrame to a CSV file
    results_df.to_csv(output_path, index=False)
    
    logging.info(f"\nPer-file accuracy results saved to: {output_path}")
    # --- END OF ADDED SECTION ---

    # Display the overall results
    logging.info("\n--- Overall Evaluation Results ---")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Which model would you like to evaluate?")
    print("[1] Contact Detection")
    print("[2] Contact Localization")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == '1':
        evaluate_best_model('contact_detection')
    elif choice == '2':
        evaluate_best_model('contact_localization')
    else:
        print("Invalid choice.")