from orchestrator import Orchestrator, ContactDetectorAI
from robot_implementations import UR5_Robot, FrankaRobot
from json_loader import JsonLoader # Import the new generic loader

if __name__ == "__main__":
    """
    Main entry point for the robot task interpreter application.
    This script is now fully configuration-driven, loading all
    behaviors and tasks from external JSON files.
    """
    print("--- Initializing System ---")
    
    # --- 1. Initialize Core Components ---
    contact_ai = ContactDetectorAI(threshold=7.5)
    
    # --- 2. Select Robot Implementation ---
    my_robot = UR5_Robot(ip_address="192.168.1.102")
    # my_robot = FrankaRobot(ip_address="192.168.1.15")

    # --- 3. Load Default Contact Behaviors from JSON ---
    json_loader = JsonLoader()
    default_behaviors = json_loader.load('src/json/AI_behaviors/default_bahavior.json')
    
    # If loading fails, proceed with an empty dictionary as a safe fallback.
    if not default_behaviors:
        print("WARNING: Could not load default behaviors. Continuing with no defaults.")
        default_behaviors = {}

    # --- 4. Configure and Run the Orchestrator ---
    controller = Orchestrator(
        robot=my_robot, 
        ai_model=contact_ai, 
        default_contact_actions=default_behaviors
    )
    
    # Specify the task file to execute.
    controller.load_task_from_file('src/json/robot_tasks/my_assembly_task.json')
    
    # Start the entire process.
    controller.run()
    
    print("--- System Shutdown ---")