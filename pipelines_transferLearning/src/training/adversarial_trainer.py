import torch
import torch.nn as nn
import torch.optim as optim
from itertools import cycle
from tqdm import tqdm
import math
import os

class AdversarialTrainer:
    """
    An improved trainer class for DANN that tracks accuracy, saves the best model,
    and supports dynamic lambda for the domain loss.
    """
    def __init__(self,
                 model: nn.Module,
                 source_loader: torch.utils.data.DataLoader,
                 target_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 task_criterion: nn.Module,
                 device: torch.device,
                 model_save_path: str = "best_model.pth",
                 use_dynamic_lambda: bool = True):
        """
        Args:
            model (nn.Module): The DANN-enabled model.
            source_loader, target_loader, val_loader (DataLoader): Data loaders.
            optimizer (Optimizer): The optimizer.
            task_criterion (nn.Module): Loss function for the main task.
            device (torch.device): 'cuda' or 'cpu'.
            model_save_path (str): Path to save the best performing model.
            use_dynamic_lambda (bool): If True, gradually increases domain loss weight.
        """
        self.model = model.to(device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.task_criterion = task_criterion
        # The domain criterion is almost always BCEWithLogitsLoss
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.device = device
        self.model_save_path = model_save_path
        self.use_dynamic_lambda = use_dynamic_lambda
        
        self.best_val_accuracy = 0.0
        self.len_dataloader = len(self.source_loader)

    def _calculate_lambda(self, epoch: int, total_epochs: int, step: int):
        """Calculates the dynamic lambda value."""
        p = (epoch * self.len_dataloader + step) / (total_epochs * self.len_dataloader)
        gamma = 10.0
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

    def _train_epoch1(self, epoch: int, total_epochs: int):
        """Performs one full training epoch."""
        self.model.train()
        total_task_loss, total_domain_loss = 0.0, 0.0
        target_iter = cycle(self.target_loader)
        
        progress_bar = tqdm(enumerate(self.source_loader), total=self.len_dataloader,
                            desc=f"Epoch {epoch}/{total_epochs} [Train]")

        for i, (source_data, source_labels) in progress_bar:
            target_data, _ = next(target_iter)
            
            # Update lambda if using dynamic scheduling
            lambda_domain = self._calculate_lambda(epoch, total_epochs, i) if self.use_dynamic_lambda else 1.0

            # Data Preparation
            source_data = source_data.to(self.device)
            target_data = target_data.to(self.device)
            combined_data = torch.cat([source_data, target_data], dim=0)

            domain_labels = torch.cat([
                torch.zeros(source_data.shape[0], device=self.device),
                torch.ones(target_data.shape[0], device=self.device)
            ])

            # Forward & Backward Pass
            self.optimizer.zero_grad()
            class_logits, domain_logits = self.model(combined_data)
            
            # Prepare labels based on task
            if self.model.task == 'localization':
                source_labels = (source_labels - 1).long().to(self.device)
            else: # detection
                source_labels = source_labels.float().to(self.device)

            task_loss = self.task_criterion(class_logits[:source_data.shape[0]], source_labels)
            domain_loss = self.domain_criterion(domain_logits, domain_labels)
            total_loss = task_loss + (lambda_domain * domain_loss)
            
            total_loss.backward()
            self.optimizer.step()

            # Logging
            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            progress_bar.set_postfix(task_loss=f"{task_loss.item():.4f}", domain_loss=f"{domain_loss.item():.4f}", lambda_d=f"{lambda_domain:.2f}")
        
        return total_task_loss / self.len_dataloader, total_domain_loss / self.len_dataloader

    def _train_epoch(self, epoch: int, total_epochs: int):
        """Performs one full training epoch."""
        self.model.train()
        total_task_loss, total_domain_loss = 0.0, 0.0
        target_iter = cycle(self.target_loader)
        
        progress_bar = tqdm(enumerate(self.source_loader), total=self.len_dataloader,
                            desc=f"Epoch {epoch}/{total_epochs} [Train]")

        for i, (source_data, source_labels) in progress_bar:
            target_data, _ = next(target_iter)
            
            # Update lambda if using dynamic scheduling
            lambda_domain = self._calculate_lambda(epoch, total_epochs, i) if self.use_dynamic_lambda else 1.0

            # Data Preparation
            source_data = source_data.to(self.device)
            target_data = target_data.to(self.device)

            domain_labels = torch.cat([
                torch.zeros(source_data.shape[0], device=self.device),
                torch.ones(target_data.shape[0], device=self.device)
            ])
            # Forward & Backward Pass
            self.optimizer.zero_grad()
            class_logits, domain_logits_s = self.model(source_data)
            _ , domain_logits_t = self.model(target_data)
            domain_logits = torch.cat([domain_logits_s, domain_logits_t], dim=0)
            
            # Prepare labels based on task
            if self.model.task == 'localization':
                source_labels = (source_labels - 1).long().to(self.device)
            else: # detection
                source_labels = source_labels.float().to(self.device)

            task_loss = self.task_criterion(class_logits[:source_data.shape[0]], source_labels)
            domain_loss = self.domain_criterion(domain_logits, domain_labels)
            total_loss = task_loss + (lambda_domain * domain_loss)
            
            total_loss.backward()
            self.optimizer.step()

            # Logging
            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            progress_bar.set_postfix(task_loss=f"{task_loss.item():.4f}", domain_loss=f"{domain_loss.item():.4f}", lambda_d=f"{lambda_domain:.2f}")
        
        return total_task_loss / self.len_dataloader, total_domain_loss / self.len_dataloader
    
    def _validate_epoch(self, epoch: int, total_epochs: int):
        """Performs one full validation epoch."""
        self.model.eval()
        total_val_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val]")

        with torch.no_grad():
            for val_data, val_labels in progress_bar:
                val_data = val_data.to(self.device)
                
                class_logits, _ = self.model(val_data)
                
                # Prepare labels for loss calculation
                if self.model.task == 'localization':
                    val_labels_for_loss = (val_labels - 1).long().to(self.device)
                    predicted = torch.argmax(class_logits, dim=1)
                    total += class_logits.size(0)
                    correct += (predicted == class_logits).sum().item()
                else:
                    val_labels_for_loss = val_labels.float().to(self.device)
                    predicted = (torch.sigmoid(class_logits) > 0.5).int()
                    total += val_labels_for_loss.size(0)
                    correct += (predicted == val_labels_for_loss.int()).sum().item()
                
                loss = self.task_criterion(class_logits, val_labels_for_loss)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        accuracy = (correct / total) * 100
        return avg_val_loss, accuracy

    def save_checkpoint(self, accuracy: float):
        """Saves the model if accuracy has improved."""
        print(f"Validation accuracy improved to {accuracy:.2f}%. Saving model...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Save the model state
        torch.save(self.model.state_dict(), self.model_save_path)


    def train(self, epochs: int):
        """The main training loop."""
        for epoch in range(1, epochs + 1):
            train_task_loss, train_domain_loss = self._train_epoch(epoch, epochs)
            val_loss, val_accuracy = self._validate_epoch(epoch, epochs)

            print(
                f"\nEpoch {epoch}/{epochs} Summary | "
                f"Train Task Loss: {train_task_loss:.3f} | "
                f"Train Domain Loss: {train_domain_loss:.3f} | "
                f"Val Task Loss: {val_loss:.3f} | "
                f"Val Accuracy: {val_accuracy:.2f}%\n"
            )

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.save_checkpoint(val_accuracy)
        
        print(f"Training finished. Best model saved at '{self.model_save_path}' with {self.best_val_accuracy:.2f}% accuracy.")