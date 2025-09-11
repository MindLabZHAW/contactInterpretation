from abc import ABC, abstractmethod

class RobotInterface(ABC):
    """
    An abstract interface (or "contract") for all robot controllers.

    This class defines a set of common methods that any robot implementation
    must have. This ensures that the high-level Orchestrator can control
    different types of robots without knowing their specific details.
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
            dict: A dictionary of sensor data, e.g., {"force_z": 10.5}.
        """
        pass

    @abstractmethod
    def send_move_command(self, move_data: dict) -> None:
        """
        Sends a non-blocking movement command to the robot. The orchestrator
        will use is_moving() to check for its completion.
        
        Args:
            move_data (dict): The data describing the next move, typically
                              from a step in the JSON task file.
        """
        pass

    @abstractmethod
    def is_moving(self) -> bool:
        """
        Checks if the robot is currently executing a move command. This is
        essential for the orchestrator's real-time monitoring loop.

        Returns:
            bool: True if the robot is busy, False if it is idle.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Sends an immediate and high-priority stop command to the robot."""
        pass