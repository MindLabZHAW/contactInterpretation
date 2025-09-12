# src/plotting.py
import matplotlib.pyplot as plt

def plot_results(ground_truth: list, predictions: list, title: str = "Model Prediction vs. Ground Truth"):
    """
    Generates and displays a plot comparing the model's predictions
    against the ground truth labels over time.
    """
    if not ground_truth or not predictions:
        print("Warning: Cannot plot results. Ground truth or predictions list is empty.")
        return

    plt.figure(figsize=(15, 6))
    
    # Plot ground truth as a solid line
    plt.plot(ground_truth, label='Ground Truth', color='blue', linestyle='-', drawstyle='steps-post')
    
    # Plot predictions as a dashed line
    plt.plot(predictions, label='Model Prediction', color='red', linestyle='--', drawstyle='steps-post')
    
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Label (1 = Contact, 0 = No Contact)")
    plt.yticks([0, 1]) # Ensure y-axis only shows 0 and 1
    plt.ylim(-0.1, 1.1) # Set limits to make the lines clear
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    print("\nDisplaying plot. Close the plot window to exit the program.")
    plt.show()