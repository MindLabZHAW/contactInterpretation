import torch
from tqdm import tqdm
import os
import logging

class DMLTrainer:
    """
    Trains a model end-to-end using a combined loss function:
    L_total = ClassificationLoss + (weight * TripletLoss)
    """
    def __init__(self, model, optimizer, classification_criterion, triplet_criterion, 
                 device, triplet_loss_weight=0.5, scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.classification_criterion = classification_criterion
        self.triplet_criterion = triplet_criterion
        self.device = device
        self.scheduler = scheduler
        self.triplet_loss_weight = triplet_loss_weight
        self.best_val_ACC = 0.0

    def _train_epoch(self, train_loader):
        self.model.train()
        total_class_loss = 0.0
        total_triplet_loss = 0.0

        train_pbar = tqdm(train_loader, desc="[Train Combined]", unit="batch")
        for anchor, positive, negative, anchor_labels in train_pbar:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            anchor_labels = anchor_labels.to(self.device).long() # For CrossEntropyLoss
            if self.model.task == 'localization':
                anchor_labels -=1
            self.optimizer.zero_grad()

            # A single forward pass on the anchor gets us both outputs
            anchor_logits, anchor_embedding = self.model(anchor)
            
            # We only need the embeddings for positive and negative samples
            _, positive_embedding = self.model(positive)
            _, negative_embedding = self.model(negative)

            # --- Calculate Losses ---
            classification_loss = self.classification_criterion(anchor_logits, anchor_labels)
            triplet_loss = self.triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            # --- Combine Losses ---
            total_loss = classification_loss + self.triplet_loss_weight * triplet_loss
            
            total_loss.backward()
            self.optimizer.step()

            total_class_loss += classification_loss.item()
            total_triplet_loss += triplet_loss.item()
            train_pbar.set_postfix(class_loss=f"{classification_loss.item():.3f}", triplet_loss=f"{triplet_loss.item():.3f}")

        return total_class_loss / len(train_loader), total_triplet_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        """Validation is performed only on the primary classification task."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="[Validation]", unit="batch")
            for inputs, _, _, labels in val_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                if self.model.task == 'localization':
                    labels -= 1
                logits, _ = self.model(inputs) # We ignore the embedding during validation
                
                loss = self.classification_criterion(logits, labels)
                val_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        return avg_val_loss, val_accuracy

    def fit(self, train_loader, val_loader, n_epochs, model_save_path, early_stop_patience=5):
        patience_counter = 0
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        for epoch in range(n_epochs):
            class_loss, triplet_loss = self._train_epoch(train_loader)
            val_loss, val_ACC = self._validate_epoch(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{n_epochs} | LR: {current_lr:.6f} | Class Loss: {class_loss:.4f} | Triplet Loss: {triplet_loss:.4f} | Val Loss: {val_loss:.4f} | Val ACC: {val_ACC:.2f}%")

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

        final_save_path = model_save_path.replace('.pth', f'_acc{self.best_val_ACC:.2f}.pth')

        if os.path.exists(model_save_path):
            os.rename(model_save_path, final_save_path)
            logging.info(f"Final model saved as: {final_save_path}")

        logging.info(f"Training finished. Best model saved to: {final_save_path}")
