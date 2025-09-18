import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import time
import multiprocessing as mp
import signal # Import the signal module

def plot_results(
    ground_truth: List[int], 
    predictions: List[int], 
    smoothed_predictions: Optional[List[int]] = None, 
    localization_predictions: Optional[List[int]] = None, 
    title: str = "AI Predictions vs. Ground Truth",
    robot_name: str = "Robot"
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
    fig.suptitle(f'{title}: {robot_name}', fontsize=16)
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

def _run_plotting_process(queue: mp.Queue, window_size: int, refresh_rate_hz: int, robot_name: str):
    """
    This function runs in a separate process. It creates and manages the plot.
    THE FIX: It ignores KeyboardInterrupts and waits for a sentinel value.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore Ctrl+C in this process
    
    plt.ion()
    fig, ax1 = plt.subplots(figsize=(15, 7))
    fig.suptitle(f"Real-Time Contact Interpretation: {robot_name}", fontsize=16)
    ax2 = ax1.twinx()

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Contact Detection", color='black')
    ax2.set_ylabel("Contact Link", color='purple')
    ax2.set_ylim(0, 7)

    line_gt, = ax1.plot([], [], label='Ground Truth', color='blue', linestyle='-', drawstyle='steps-post')
    line_pred, = ax1.plot([], [], label='Detection Prediction', color='red', linestyle='--', drawstyle='steps-post', alpha=0.6)
    line_smooth, = ax1.plot([], [], label='Smoothed Detection', color='green', linestyle='-', drawstyle='steps-post', linewidth=2)
    scatter_loc = ax2.scatter([], [], label='Localization Prediction', color='purple', marker='x', s=50)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    x_data, gt_data, pred_data, smooth_data, loc_data = [], [], [], [], []
    
    last_draw_time = 0
    draw_interval = 1.0 / refresh_rate_hz

    while True:
        try:
            # Check for the sentinel value to terminate
            data_point = queue.get()
            if data_point is None:
                plt.close(fig)
                return

            x_data.append(data_point[0])
            gt_data.append(data_point[1])
            pred_data.append(data_point[2])
            smooth_data.append(data_point[3])
            loc_data.append(data_point[4])

            if time.time() - last_draw_time < draw_interval:
                continue
            
            last_draw_time = time.time()
            
            x, gt, pred, smooth, loc = (d[-window_size:] for d in [x_data, gt_data, pred_data, smooth_data, loc_data])

            if not x: continue

            line_gt.set_data(x, gt)
            line_pred.set_data(x, pred)
            line_smooth.set_data(x, smooth)
            
            valid_loc_x = [xi for i, xi in enumerate(x) if loc[i] != -1]
            valid_loc_y = [yi for yi in loc if yi != -1]
            scatter_loc.set_offsets(np.c_[valid_loc_x, valid_loc_y])

            ax1.relim()
            ax1.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        except Exception:
            # Catch other potential errors to prevent the process from crashing
            continue

class RealTimePlotter:
    """
    A process-safe plotter that runs matplotlib in a separate process
    to prevent blocking the main application.
    """
    def __init__(self, window_size: int = 200, refresh_rate_hz: int = 10, robot_name: str = "Robot"):
        self.queue = mp.Queue()
        self.plot_process = mp.Process(
            target=_run_plotting_process, 
            args=(self.queue, window_size, refresh_rate_hz, robot_name)
        )
        self.step_count = 0

    def start(self):
        """Starts the plotting process."""
        self.plot_process.start()

    def update(self, gt: int, pred: int, smooth_pred: int, loc_pred: int):
        """Sends data to the plotting process. This is a non-blocking call."""
        self.step_count += 1
        try:
            self.queue.put_nowait((self.step_count, gt, pred, smooth_pred, loc_pred))
        except Exception:
            # Queue might be full or closed, just ignore
            pass

    def close(self):
        """Signals the plotting process to terminate and waits for it."""
        if self.plot_process.is_alive():
            self.queue.put(None) # Send the sentinel value
            self.plot_process.join(timeout=2) # Wait for the process to finish
            if self.plot_process.is_alive():
                self.plot_process.terminate() # Forcefully stop if it doesn't close