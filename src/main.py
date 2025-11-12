import logging
import socket
import sys
import os # <-- Added for path joining
from robot_control.task_interpreter import TaskInterpreter
from contact_interpretation.interpreter import ContactAI
from robot_control import robots 
from config_loader import ConfigLoader
from data_logger import DataLogger

# --- Constants for Config Paths ---
# Use os.path.join to create OS-agnostic paths
SRC_DIR = os.path.dirname(__file__)
MAIN_CONFIG_FILE = os.path.join(SRC_DIR, 'config', 'config.yaml')
DEFAULT_BEHAVIORS_FILE = os.path.join(SRC_DIR, 'config', 'default_behaviors.yaml')


def select_robot_profile(config: dict) -> str:
    """
    Displays a menu of available robot profiles and prompts the user to choose one.
    """
    # Reads from "robot_profiles" instead of "robot_settings"
    profile_options = list(config.get("robot_profiles", {}).keys())
    if not profile_options:
        logging.error("No 'robot_profiles' found in your config.yaml.")
        return None

    print("\n--- Please Select a Robot Profile ---")
    for i, name in enumerate(profile_options):
        print(f"[{i + 1}] {name}")
    
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(profile_options)}): "))
            if 1 <= choice <= len(profile_options):
                return profile_options[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def discover_active_robot(config: dict) -> str:
    """
    Tries to connect to each robot in 'robot_profiles' to find an active one.
    """
    robot_profiles = config.get("robot_profiles", {})
    for name, profile in robot_profiles.items():
        # Reads from "robot_init_args"
        ip = profile.get("robot_init_args", {}).get("ip_address")
        if not ip:
            continue 

        try:
            logging.info(f"Checking for robot '{name}' at {ip}...")
            # Use a standard port (e.g., 80 for web interface) for a quick check
            with socket.create_connection((ip, 80), timeout=1): 
                logging.info(f"✅ Found active robot profile: {name}")
                return name
        except (socket.timeout, ConnectionRefusedError, OSError):
            logging.info(f"No response from '{name}'.")
            continue
            
    return None


def create_robot_from_profile(config: dict, active_profile_name: str):
    """
    Factory function to create a robot instance and get all paths from a profile.
    
    Returns a tuple of:
    (robot_instance, task_file_path, loop_delay, selected_features, ai_model_config_path)
    """
    # Reads from "robot_profiles"
    profile = config.get("robot_profiles", {}).get(active_profile_name)

    if not profile:
        logging.error(f"Configuration for active profile '{active_profile_name}' not found.")
        return None, None, None, None, None

    class_name = profile.get("robot_class")
    init_args = profile.get("robot_init_args", {})
    selected_features = init_args.get("selected_features")
    
    try:
        RobotClass = getattr(robots, class_name)
        robot_instance = RobotClass(**init_args)
    except AttributeError:
        logging.error(f"Robot class '{class_name}' not found in 'robot_control.robots' module.")
        return None, None, None, None, None
    except Exception as e:
        logging.error(f"Failed to instantiate robot '{class_name}': {e}")
        return None, None, None, None, None

    # Get all the other settings from the profile
    task_file = profile.get("task_file")
    loop_delay = profile.get("loop_delay")
    ai_model_config_path = profile.get("ai_model_config")
    
    return robot_instance, task_file, loop_delay, selected_features, ai_model_config_path

if __name__ == "__main__":
    """
    Main entry point for the robot contact interpretation application.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Initializing System ---")
    
    config_loader = ConfigLoader()
    
    # --- 1. Load Main Config (YAML) ---
    config = config_loader.load_yaml(MAIN_CONFIG_FILE)
    if not config:
        logging.error(f"Could not load {MAIN_CONFIG_FILE}. Exiting application.")
        sys.exit(1)

    # --- 2. Determine Active Profile ---
    active_profile_name = config.get("active_robot_profile")
    
    if active_profile_name == "INTERACTIVE":
        active_profile_name = select_robot_profile(config)
        if not active_profile_name:
            sys.exit(1)
    elif active_profile_name == "AUTO_DETECT":
        active_profile_name = discover_active_robot(config)
        if not active_profile_name:
            logging.error("Auto-detect failed: No configured robots found on the network.")
            sys.exit(1)
    elif active_profile_name not in config.get("robot_profiles", {}):
        logging.error(f"The specified active_robot_profile '{active_profile_name}' does not exist in the configuration.")
        sys.exit(1)
        
    logging.info(f"--- 🚀 Activating Profile: {active_profile_name} ---")

    # --- 3. Create Robot and Get Settings ---
    my_robot, task_file, loop_delay, selected_features, ai_config_path = create_robot_from_profile(config, active_profile_name)
    
    if my_robot is None:
        sys.exit(1)
    
    if not all([task_file, selected_features, ai_config_path]):
        logging.error("Profile is missing one or more required keys: 'task_file', 'selected_features' (in robot_init_args), or 'ai_model_config'.")
        sys.exit(1)

    NUM_FEATURES = len(selected_features)

    # --- 4. Load AI Model Config (YAML) ---
    ai_config = config_loader.load_yaml(ai_config_path)
    if not ai_config:
        logging.error(f"Could not load AI model config: {ai_config_path}. Exiting.")
        sys.exit(1)
    
    # --- 5. Initialize AI ---
    # We will update ContactAI to accept ai_config in its constructor
    contact_ai = ContactAI(ai_model_config=ai_config, num_features=NUM_FEATURES)
    
    # This check will be updated inside the new ContactAI
    # if contact_ai.detection_model is None or contact_ai.localization_model is None:
    #     logging.error("Model loading failed. Exiting application.")
    #     sys.exit(1)

    # --- 6. Initialize Logger ---
    data_logger = None
    save_data = input("Do you want to save the collected data to a CSV file? (y/n): ").lower()
    if save_data == 'y':
        # Headers are now dynamic from the profile
        log_headers = ['label','raw_pred', 'smoothed_pred', 'loc_pred'] + selected_features
        data_logger = DataLogger(headers=log_headers)

    # --- 7. Load Default Behaviors (YAML) ---
    default_behaviors = config_loader.load_yaml(DEFAULT_BEHAVIORS_FILE)
    if not default_behaviors:
        logging.warning(f"Could not load {DEFAULT_BEHAVIORS_FILE}. Continuing with no defaults.")
        default_behaviors = {}

    # --- 8. Initialize and Run Interpreter ---
    controller = TaskInterpreter(
        robot=my_robot, 
        ai_model=contact_ai, 
        default_contact_actions=default_behaviors,
        data_logger=data_logger,
        loop_delay=loop_delay,
        realtime_plot=True
    )
    
    controller.load_task_from_file(task_file) # Use the task_file from the profile
    controller.run()
    
    logging.info("--- System Shutdown ---")
    sys.exit(0)