import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def majority_voting_last_n(model_out, n):
    """
    Apply majority voting for each step considering the last `n` items.

    Args:
        model_out (pd.Series): Predicted classes for each step.
        n (int): Number of previous items (including the current one) to consider for voting.

    Returns:
        pd.Series: Smoothed predictions based on majority voting.
    """
    smoothed_predictions = model_out.copy()  # Copy to retain index
    for i in range(n, len(model_out), 1):
        # Define the window
        start_idx = i - n 
        end_idx = i  # Include the current item
        window = model_out.iloc[start_idx:end_idx]
        
        # Perform majority voting
        most_common = Counter(window).most_common(1)[0][0]
        smoothed_predictions.iloc[i-1] = most_common
    
    return smoothed_predictions

def calculate_metrics(y_true, y_pred, is_binary=True):
    """Calculates metrics for binary or multi-class tasks."""
    if is_binary:
        # Your existing binary metric calculations
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return accuracy, precision, recall, f1
    else:
        # Metrics for multi-class classification
        accuracy = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        return accuracy, f1_micro
    


def contact_detection_accuracy(df, allowable_delay = 20):
    "df should include label, majority voting and time"
    TP, TN, FP, FN = 0, 0, 0, 0
    
    df['label_diff'] = df['label'].diff().fillna(0)
    df['TP'], df['TN'], df['FP'], df['FN'] = None, None, None, None

    contact_delays = []
    no_contact_delays = []
    df = df.reset_index(drop=True)

    change_events = df[df['label_diff'] != 0].index.tolist()

    # Add first and last index if not present
    if 0 not in change_events:
        change_events.insert(0, 0)
    if len(df) - 1 not in change_events:
        change_events.append(len(df) - 1)

    for event_idx in range(len(change_events) - 1):
        idx = change_events[event_idx]
        
        if df.label[idx] == 1: # Starting a contact event
            for i in range(idx, min(idx + allowable_delay, len(df))):
                if df.majority_voting[i] == 1:
                    delay = df.time[i] - df.time[idx]
                    contact_delays.append(delay)
                    break
            
            if i >= (idx + allowable_delay - 1):
                contact_delays.append(0)

            # Calculate TP and FN
            for j in range(change_events[event_idx+1], max(change_events[event_idx+1]-allowable_delay, 0), -1):
                if df.majority_voting[j] == 1:
                    break

            for idx_ in range(i, j):
                if df.majority_voting[idx_] == df.label[idx_]:
                    TP += 1
                    df.loc[idx_, 'TP'] = 0.9
                else:
                    FN += 1
                    df.loc[idx_, 'FN'] = 0.9

        else: # Starting a contact-free event

            for i in range(idx, min(idx + allowable_delay, len(df))):
                if df.majority_voting[i] == 0:
                    delay = df.time[i] - df.time[idx]
                    no_contact_delays.append(delay)
                    break
            
            if i >= (idx + allowable_delay - 1):
                no_contact_delays.append(0)

            # Calculate TN and FP
            for j in range(change_events[event_idx+1], max(change_events[event_idx+1]-allowable_delay, 0), -1):
                if df.majority_voting[j] == 0:
                    break
                
            for idx_ in range(i, j):
                if df.majority_voting[idx_] == df.label[idx_]:
                    TN += 1
                    df.loc[idx_, 'TN'] = 0.1
                else:
                    FP += 1
                    df.loc[idx_, 'FP'] = 0.1

    # Compute averages safely
    contact_avg_delay = np.nanmean([d for d in contact_delays if d > 0])
    no_contact_avg_delay = np.nanmean([d for d in no_contact_delays if d > 0])

    ModelAccuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    DetectionFailureRate = FN / (TP + FN) * 100
    FalseAlarmRate = FP / (TN + FP) * 100

    return df, TP, TN, FP, FN, contact_avg_delay, no_contact_avg_delay, ModelAccuracy, DetectionFailureRate, FalseAlarmRate, contact_delays, no_contact_delays

# Call the function
df, TP, TN, FP, FN, contact_avg_delay, no_contact_avg_delay, ModelAccuracy, DetectionFailureRate, FalseAlarmRate, contact_delays, no_contact_delays = contact_detection_accuracy(df, 20)

print('contact_delays:', contact_delays)
print('no_contact_delays:', no_contact_delays)
print(f"Accuracy:{ModelAccuracy:.2f}%, DetectionFailureRate:{DetectionFailureRate:.2f}%, FalseAlarmRate:{FalseAlarmRate:.2f}%, Contact Detection Delay: {contact_avg_delay*1000:.3f}ms, No-Contact Detection Delay: {no_contact_avg_delay*1000:.3f}ms, TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Plot results
df[['time', 'label', 'majority_voting', 'TP', 'TN', 'FP', 'FN']].iplot(
    x='time', kind='scatter', 
    mode={'label': 'lines', 'majority_voting': 'lines', 'TP': 'markers', 'TN': 'markers', 'FP': 'markers', 'FN': 'markers'},
    title='Contact Detection Analysis (Interactive)',
    xTitle='Time Index', yTitle='Signal', theme='white',
    symbol=[None, None, 'x', 'x', 'x', 'x'],
    colors=['blue', 'pink', 'green', 'lightgreen', 'orange', 'red'],
)
