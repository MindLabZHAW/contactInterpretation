import torch
from tqdm import tqdm
import os
import logging
import math

class DMLTrainer:
    """
    Trains a model in two sequential stages (Embedding, then Classifier)
    using a single .fit() call.
    
    This trainer uses a fixed (hard-coded) configuration for the
    two-stage process. Both stages include validation and early stopping.
    """
    def __init__(self, 
                 model, 
                 optimizer,  # This will be used for Stage 1
                 classification_criterion, 
                 triplet_criterion, 
                 device, 
                 triplet_loss_weight=0.5, # This will be IGNORED
                 scheduler=None):         # This will be used for Stage 1
        """
        Initializes the two-stage trainer.

        Args:
            model: The neural network model.
            optimizer: The optimizer for Stage 1 (e.g., Adam(model.parameters(), lr=1e-4)).
            classification_criterion: Loss function for classification.
            triplet_criterion: Loss function for embeddings.
            device: The device to run on (e.g., 'cuda' or 'cpu').
            triplet_loss_weight: (IGNORED) This is not used in two-stage training.
            scheduler: (Optional) The scheduler for the Stage 1 optimizer.
        """
        self.model = model.to(device)
        self.classification_criterion = classification_criterion
        self.triplet_criterion = triplet_criterion
        self.device = device
        
        # --- Stage 1 Config (from __init__ args) ---
        self.optimizer_emb = optimizer
        self.scheduler_emb = scheduler
        self. optimizer_cls = optimizer
        # --- Hard-coded Two-Stage Parameters ---
        self.cls_learning_rate = 1e-2
        self.scheduler_cls_fn = scheduler #lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2)
        self.freeze_backbone_fn = self._default_freeze_backbone
        # --- End of Hard-coded Parameters ---

        if triplet_loss_weight != 0.5:
             logging.warning(f"DMLTrainer: 'triplet_loss_weight' ({triplet_loss_weight}) is set but will be IGNORED in two-stage training mode.")

        self.best_val_ACC = 0.0
        self.best_val_triplet_loss = float('inf')
        self.has_task_attribute = hasattr(self.model, 'task')

    def _default_freeze_backbone(self, model_to_freeze):
        """
        A hard-coded function to try and freeze the model's backbone.
        It checks for common attribute names.
        """
        # --- UPDATED THIS LINE ---
        # Added 'embedding_net' to match your DMLClassificationNet model
        common_backbone_names = ['backbone', 'features', 'conv_base', 'base_model', 'embedding_net']
        found_backbone = False
        
        for name in common_backbone_names:
            if hasattr(model_to_freeze, name):
                logging.info(f"Found backbone attribute: '{name}'. Freezing its parameters.")
                backbone = getattr(model_to_freeze, name)
                
                # Freeze all parameters in the found module
                for param in backbone.parameters():
                    param.requires_grad = False
                
                found_backbone = True
                break
        
        if not found_backbone:
            logging.warning("Could not find a common backbone attribute (e.g., 'backbone', 'features', 'embedding_net').")
            logging.warning("Stage 2 will fine-tune the ENTIRE model.")

    # --- Internal Epoch Methods ---

    def _train_epoch_embedding(self, train_loader):
        """(Stage 1) Trains one epoch using triplet loss only."""
        self.model.train()
        total_triplet_loss = 0.0
        train_pbar = tqdm(train_loader, desc="[Train Embedding]", unit="batch")
        
        for anchor, positive, negative, _ in train_pbar:
            anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
            
            self.optimizer_emb.zero_grad()
            # We call self.model, which is the DMLClassificationNet
            # It returns (logits, embedding). We only need the embedding.
            _, anchor_embedding = self.model(anchor)
            _, positive_embedding = self.model(positive)
            _, negative_embedding = self.model(negative)

            triplet_loss = self.triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
            triplet_loss.backward()
            self.optimizer_emb.step()

            total_triplet_loss += triplet_loss.item()
            train_pbar.set_postfix(triplet_loss=f"{triplet_loss.item():.3f}")
        return total_triplet_loss / len(train_loader)

    def _train_epoch_classifier(self, train_loader, optimizer):
        """(Stage 2) Trains one epoch using classification loss only."""
        self.model.train() # Model is in train mode, but backbone is frozen
        total_class_loss = 0.0
        train_pbar = tqdm(train_loader, desc="[Train Classifier]", unit="batch")

        for anchor, _, _, anchor_labels in train_pbar:
            anchor, anchor_labels = anchor.to(self.device), anchor_labels.to(self.device).long()
            
            if self.has_task_attribute and self.model.task == 'localization':
                anchor_labels -= 1
            
            optimizer.zero_grad()
            # We only need the logits (output_logits)
            anchor_logits, _ = self.model(anchor) 
            
            classification_loss = self.classification_criterion(anchor_logits, anchor_labels)
            classification_loss.backward()
            optimizer.step()

            total_class_loss += classification_loss.item()
            train_pbar.set_postfix(class_loss=f"{classification_loss.item():.3f}")
        return total_class_loss / len(train_loader)

    def _validate_epoch_embedding(self, val_loader):
        """(Stage 1) Validation on the triplet loss task."""
        self.model.eval()
        total_val_triplet_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="[Val Embedding]", unit="batch")
            for anchor, positive, negative, _ in val_pbar:
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                
                _, anchor_embedding = self.model(anchor)
                _, positive_embedding = self.model(positive)
                _, negative_embedding = self.model(negative)

                triplet_loss = self.triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
                total_val_triplet_loss += triplet_loss.item()
        
        return total_val_triplet_loss / len(val_loader)

    def _validate_epoch_classifier(self, val_loader):
        """(Stage 2) Validation on the classification task."""
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="[Val Classifier]", unit="batch")
            # Assumes val_loader yields (inputs, pos, neg, labels)
            for inputs, _, _, labels in val_pbar: 
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                if self.has_task_attribute and self.model.task == 'localization':
                    labels -= 1
                
                logits, _ = self.model(inputs) # Get logits
                loss = self.classification_criterion(logits, labels)
                val_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        return avg_val_loss, val_accuracy

    # --- Public Fit Method ---

    def fit(self, train_loader, val_loader, n_epochs, model_save_path, early_stop_patience=5):
        """
        Runs the full two-stage training process.
        """
        
        # --- 0. Calculate Epochs and Save Paths ---
        n_epochs_emb = n_epochs
        n_epochs_cls = n_epochs 
        if n_epochs_emb == 0 or n_epochs_cls == 0:
            logging.warning(f"Epoch split resulted in 0 epochs for one stage. Stage 1: {n_epochs_emb}, Stage 2: {n_epochs_cls}.")
            if n_epochs_emb == 0 and n_epochs_cls > 0: logging.warning("Skipping Stage 1.")
            elif n_epochs_cls == 0 and n_epochs_emb > 0: logging.warning("Skipping Stage 2.")
            else: logging.error("No epochs to train."); return

        base_dir = os.path.dirname(model_save_path)
        base_name = os.path.basename(model_save_path)
        
        if '.' in base_name:
            base_name_no_ext = base_name.rsplit('.', 1)[0]
            ext = base_name.rsplit('.', 1)[1]
        else:
            base_name_no_ext = base_name
            ext = 'pth'
            
        model_save_path_emb = os.path.join(base_dir, f"{base_name_no_ext}_stage1_emb.{ext}")
        model_save_path_cls = os.path.join(base_dir, f"{base_name_no_ext}_stage2_cls.{ext}")

        logging.info(f"Total Epochs: {n_epochs} | Stage 1 (Embedding): {n_epochs_emb} epochs | Stage 2 (Classifier): {n_epochs_cls} epochs")

        # ---
        # STAGE 1: EMBEDDING TRAINING
        # ---
        best_stage1_model_path = ""
        if n_epochs_emb > 0:
            logging.info("--- Starting Stage 1: Embedding Training ---")
            patience_counter_emb = 0
            self.best_val_triplet_loss = float('inf')
            
            base_save_path_emb = model_save_path_emb.replace(f'.{ext}', f'_best.{ext}')
            os.makedirs(os.path.dirname(base_save_path_emb), exist_ok=True)
            
            for epoch in range(n_epochs_emb):
                train_triplet_loss = self._train_epoch_embedding(train_loader)
                val_triplet_loss = self._validate_epoch_embedding(val_loader)
                
                current_lr = self.optimizer_emb.param_groups[0]['lr']
                logging.info(f"[Stage 1] Epoch {epoch+1}/{n_epochs_emb} | LR: {current_lr:.6f} | Train Triplet Loss: {train_triplet_loss:.4f} | Val Triplet Loss: {val_triplet_loss:.4f}")

                if self.scheduler_emb:
                    if isinstance(self.scheduler_emb, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler_emb.step(val_triplet_loss) # Step with validation loss
                    else:
                        self.scheduler_emb.step()

                if val_triplet_loss < self.best_val_triplet_loss:
                    self.best_val_triplet_loss = val_triplet_loss
                    torch.save(self.model.state_dict(), base_save_path_emb)
                    logging.info(f"*** New best embedding model saved with Val Triplet Loss: {self.best_val_triplet_loss:.4f} ***")
                    best_stage1_model_path = base_save_path_emb # Store path to best model
                    patience_counter_emb = 0
                else:
                    patience_counter_emb += 1

                if patience_counter_emb >= early_stop_patience:
                    logging.info("Early stopping triggered for embedding stage.")
                    break

            final_save_path_emb = base_save_path_emb.replace(f'_best.{ext}', f'_final_loss{self.best_val_triplet_loss:.4f}.{ext}')
            if os.path.exists(base_save_path_emb):
                os.rename(base_save_path_emb, final_save_path_emb)
                best_stage1_model_path = final_save_path_emb # Update path to renamed file
                logging.info(f"Final embedding model saved as: {final_save_path_emb}")
            else:
                logging.warning(f"No best embedding model was saved (path not found: {base_save_path_emb}).")

            logging.info(f"--- Embedding stage finished. Best Val Triplet Loss: {self.best_val_triplet_loss:.4f} ---")
            
            # ** CRITICAL: Load the best model from stage 1 before proceeding **
            if best_stage1_model_path and os.path.exists(best_stage1_model_path):
                logging.info(f"Loading best embedding model from: {best_stage1_model_path}")
                self.model.load_state_dict(torch.load(best_stage1_model_path, map_location=self.device))
            else:
                logging.warning("Could not find best embedding model to load. Proceeding with last epoch's weights.")

        # ---
        # INTERMISSION: FREEZE BACKBONE & CREATE NEW OPTIMIZER
        # ---
        logging.info("--- Preparing for Stage 2: Freezing Backbone ---")
        self.freeze_backbone_fn(self.model)

        logging.info("Creating new optimizer for classifier head...")
        # Create new optimizer for *only* the parameters that still require gradients
        optimizer_cls = self.optimizer_cls
        #torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cls_learning_rate)
        
        trainable_params = [name for name, p in self.model.named_parameters() if p.requires_grad]
        logging.info(f"Classifier optimizer will train: {trainable_params}")
        
        scheduler_cls = self.scheduler_cls_fn#(optimizer_cls) if self.scheduler_cls_fn else None

        # ---
        # STAGE 2: CLASSIFIER TRAINING
        # ---
        if n_epochs_cls > 0:
            logging.info("--- Starting Stage 2: Classifier Training ---")
            patience_counter_cls = 0
            self.best_val_ACC = 0.0 # Reset best accuracy
            
            base_save_path_cls = model_save_path_cls.replace(f'.{ext}', f'_best.{ext}')
            os.makedirs(os.path.dirname(base_save_path_cls), exist_ok=True)

            for epoch in range(n_epochs_cls):
                class_loss = self._train_epoch_classifier(train_loader, optimizer_cls)
                val_loss, val_ACC = self._validate_epoch_classifier(val_loader)

                current_lr = optimizer_cls.param_groups[0]['lr']
                logging.info(f"[Stage 2] Epoch {epoch+1}/{n_epochs_cls} | LR: {current_lr:.6f} | Class Loss: {class_loss:.4f} | Val Loss: {val_loss:.4f} | Val ACC: {val_ACC:.2f}%")

                if scheduler_cls:
                    if isinstance(scheduler_cls, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler_cls.step(val_loss)
                    else:
                         scheduler_cls.step()

                if val_ACC > self.best_val_ACC:
                    self.best_val_ACC = val_ACC
                    torch.save(self.model.state_dict(), base_save_path_cls)
                    logging.info(f"*** New best classifier model saved with ACC: {self.best_val_ACC:.2f}% ***")
                    patience_counter_cls = 0
                else:
                    patience_counter_cls += 1

                if patience_counter_cls >= early_stop_patience:
                    logging.info("Early stopping triggered for classifier stage.")
                    break
            
            final_save_path_cls = base_save_path_cls.replace(f'_best.{ext}', f'_final_acc{self.best_val_ACC:.2f}.{ext}')
            
            if os.path.exists(base_save_path_cls):
                os.rename(base_save_path_cls, final_save_path_cls)
                logging.info(f"Final classifier model saved as: {final_save_path_cls}")
            else:
                logging.warning(f"No best classifier model was saved (path not found: {base_save_path_cls}).")

            logging.info(f"--- Two-stage training finished. Best classifier ACC: {self.best_val_ACC:.2f}% ---")
        
        else:
             logging.info("--- Two-stage training finished (Stage 2 skipped). ---")