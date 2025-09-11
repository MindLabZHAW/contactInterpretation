from abc import ABC, abstractmethod

class RobotInterface(ABC):
    """
    An abstract interface (or "contract") for all robot controllers.
    This class defines the common methods any robot implementation must have.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Establishes a connection to the physical robot hardware."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection to the robot."""
        pass

    @abstractmethod
    def get_data(self) -> dict:
        """Retrieves real-time sensor data from the robot."""
        pass

    @abstractmethod
    def send_move_command(self, move_data: dict) -> None:
        """Sends a non-blocking movement command to the robot."""
        pass

    @abstractmethod
    def is_moving(self) -> bool:
        """Checks if the robot is currently executing a move command."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Sends an immediate stop command to the robot."""
        pass
