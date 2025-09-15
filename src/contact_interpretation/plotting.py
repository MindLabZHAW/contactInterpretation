import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

def plot_results(
    ground_truth: List[int], 
    predictions: List[int], 
    smoothed_predictions: Optional[List[int]] = None, 
    localization_predictions: Optional[List[int]] = None, 
    title: str = "AI Predictions vs. Ground Truth"
):
    """
    Generates and displays a plot comparing predictions against ground truth labels.
    It uses a dual-axis plot to show contact detection and localization simultaneously.

    Args:
        ground_truth (List[int]): The actual contact labels from the simulation.
        predictions (List[int]): The raw predictions from the contact detection model.
        smoothed_predictions (Optional[List[int]]): Smoothed detection predictions after majority voting.
        localization_predictions (Optional[List[int]]): The predicted contact link from the localization model.
        title (str): The main title for the plot.
    """
    if not ground_truth or not predictions:
        print("Warning: Cannot plot results. Ground truth or predictions list is empty.")
        return

    fig, ax1 = plt.subplots(figsize=(15, 7))

    # --- Primary Y-axis (for Contact Detection) ---
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Contact Detection (1=Contact)", color='black')
    
    # Plot the ground truth as a solid blue line
    ax1.plot(ground_truth, label='Ground Truth', color='blue', linestyle='-', drawstyle='steps-post')
    # Plot the raw model predictions as a dashed red line
    ax1.plot(predictions, label='Detection Prediction', color='red', linestyle='--', drawstyle='steps-post', alpha=0.6)
    
    # Plot the smoothed predictions if they are available
    if smoothed_predictions:
        ax1.plot(smoothed_predictions, label='Smoothed Detection', color='green', linestyle='-', drawstyle='steps-post', linewidth=2)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_yticks([0, 1])
    ax1.set_ylim(-0.1, 1.1)

    # --- Secondary Y-axis (for Contact Localization) ---
    if localization_predictions:
        ax2 = ax1.twinx() # Create a second y-axis that shares the same x-axis
        ax2.set_ylabel("Contact Link (Localization)", color='purple')
        
        # Plot localization results as a scatter plot for better visibility
        timesteps = np.arange(len(localization_predictions))
        # Filter out the -1 values so they don't get plotted when no contact is detected
        valid_loc_preds = [p for p in localization_predictions if p != -1]
        valid_timesteps = [t for i, t in enumerate(timesteps) if localization_predictions[i] != -1]

        ax2.scatter(valid_timesteps, valid_loc_preds, label='Localization Prediction', color='purple', marker='x', s=50)
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_ylim(0, 7) # Assuming 7 robot links/joints
        ax2.grid(None) # Turn off the grid for the secondary axis to reduce clutter

    # --- Final Touches ---
    fig.suptitle(title, fontsize=16)
    # Collect all labels from all axes for a single legend
    lines, labels = ax1.get_legend_handles_labels()
    if 'ax2' in locals():
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='upper right')

    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    print("\nDisplaying plot. Close the plot window to exit the program.")
    plt.show()