from abc import ABC, abstractmethod

class AbstractRobot(ABC):
    """
    An abstract interface (or "contract") for all robot controllers.

    This class defines a set of common methods that any robot implementation
    must have. This ensures that the high-level Orchestrator can control
    different types of robots without knowing their specific details. It's the
    foundation of the "plug-and-play" robot architecture.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establishes a connection to the physical robot hardware.
        
        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection to the robot."""
        pass

    @abstractmethod
    def get_data(self) -> dict:
        """
        Retrieves real-time sensor data from the robot.
        
        Returns:
            dict: A dictionary containing key-value pairs of sensor data,
                  such as {"force_z": 10.5, "joint_angles": [...]}.
        """
        pass

    @abstractmethod
    def send_move_command(self, move_data) -> None:
        """
        Sends a movement command to the robot.
        
        Args:
            move_data: The data describing the next move (e.g., target pose,
                       waypoint, or a simple continuation signal).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Sends an immediate and high-priority stop command to the robot."""
        pass