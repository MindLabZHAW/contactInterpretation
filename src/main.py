from orchestrator import Orchestrator, ContactDetectorAI
from robot_implementations import UR5_Robot, FrankaRobot

if __name__ == "__main__":
    """
    This is the main entry point for the robot control application.
    """
    
    # 1. Initialize the AI Model
    # Set the force threshold (in Newtons) for detecting contact.
    print("--- Initializing System ---")
    contact_ai = ContactDetectorAI(threshold=8)

    # 2. CHOOSE YOUR ROBOT IMPLEMENTATION 🔌
    # This is the only section you need to change to switch robots.
    # Just comment out one line and uncomment the other.
    print("Selecting robot implementation...")
    
    # --- Option A: Use the UR5 ---
    my_robot = UR5_Robot(ip_address="192.168.1.102")

    # --- Option B: Use the Franka ---
    # my_robot = FrankaRobot(ip_address="192.168.1.15")
    
    # 3. Inject the chosen robot and AI into the orchestrator
    # The Orchestrator works with any 'my_robot' 
    #  as long as it
    # follows the rules defined in the AbstractRobot class.
    print("Creating controller...")
    controller = Orchestrator(robot=my_robot, ai_model=contact_ai)
    
    # 4. Run the process
    # This connects to the robot and starts the main control loop.
    controller.run_process()
    
    print("--- System Shutdown ---")