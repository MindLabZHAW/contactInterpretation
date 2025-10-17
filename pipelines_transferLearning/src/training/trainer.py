import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from typing import Callable

class Trainer:
    """
    A universal, class-based trainer for PyTorch models.
    """
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
                 device: torch.device, scheduler: optim.lr_scheduler._LRScheduler = None):
        """
        Initializes the Trainer object.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        self.best_val_ACC = 0.0

    def _train_epoch(self, train_loader: DataLoader, use_gradient_clipping: bool) -> float:
        """Conducts a single training epoch."""
        self.model.train()
        total_loss = 0.0
        nan_skips = 0

        train_pbar = tqdm(train_loader, desc="[Train]", unit="batch")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)

            # Updated autocast to avoid FutureWarning
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                if self.model.domain_discriminator:
                    outputs, _ = self.model(inputs)
                else: 
                    outputs = self.model(inputs)

                if self.model.task == 'localization':
                    labels = (labels - 1).long()
                else:
                    labels = labels.float()
                loss = self.criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_skips += 1
                continue

            self.scaler.scale(loss).backward()

            if use_gradient_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        if nan_skips > 0:
            logging.warning(f"Skipped {nan_skips} batches due to NaN/Inf loss.")

        return total_loss / (len(train_loader) - nan_skips) if (len(train_loader) - nan_skips) > 0 else 0.0

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, float]:
        """Conducts a single validation epoch."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="[Val]", unit="batch")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                # Updated autocast to avoid FutureWarning
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    if self.model.domain_discriminator:
                        outputs, _ = self.model(inputs)
                    else: 
                        outputs = self.model(inputs)
                
                if self.model.task == 'detection':
                    val_loss += self.criterion(outputs, labels).item()
                    predicted = (torch.sigmoid(outputs) > 0.5).int()
                    total += labels.size(0)
                    correct += (predicted == labels.int()).sum().item()
                    val_loss += self.criterion(outputs, labels).item()
                else:
                    target_labels = (labels - 1).long()
                    val_loss += self.criterion(outputs, target_labels).item()
                    predicted = torch.argmax(outputs, dim=1)
                    total += target_labels.size(0)
                    correct += (predicted == target_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        return avg_val_loss, val_accuracy

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int,
            model_save_path: str, final_model_name: str, early_stop_patience: int = 5,
            use_gradient_clipping: bool = True):
        """
        Runs the full training and validation loop.
        """
        patience_counter = 0

        for epoch in range(n_epochs):
            
            train_loss = self._train_epoch(train_loader, use_gradient_clipping)
            val_loss, val_ACC = self._validate_epoch(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{n_epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ACC: {val_ACC:.2f}%")

            if self.scheduler:
                self.scheduler.step(val_loss)

            if val_ACC > self.best_val_ACC:
                self.best_val_ACC = val_ACC
                torch.save(self.model.state_dict(), model_save_path)
                logging.info(f"*** New best model saved with ACC: {self.best_val_ACC:.2f}% ***")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break
        
        base_dir = os.path.dirname(model_save_path)
        final_model_name_with_acc = final_model_name.replace('.pth', f'_acc{self.best_val_ACC:.2f}.pth')
        final_save_path = os.path.join(base_dir, final_model_name_with_acc)
        
        if os.path.exists(model_save_path):
            os.rename(model_save_path, final_save_path)
            logging.info(f"Final model saved as: {final_save_path}")