"""
Simple Trainer for MNIST Classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
import os
import matplotlib.pyplot as plt
import logging


class SimpleTrainer:
    """
    A simple trainer class that handles the training loop for our MNIST model.
    Designed to be educational and easy to understand.
    """
    
    def __init__(self, model, config):
        """
        Initialize the trainer
        
        Args:
            model: PyTorch model to train
            config (dict): Configuration dictionary
        """
        print("🚀 Initializing trainer...")
        logging.info("Initializing trainer...")
        
        self.model = model
        self.config = config
        
        # Training configuration
        self.learning_rate = config['training']['learning_rate']
        self.max_epochs = config['training']['max_epochs']
        self.device = torch.device(config['training']['device'])
        
        # Move model to device
        self.model.to(self.device)
        print(f"   Using device: {self.device}")
        logging.info(f"Using device: {self.device}")
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Checkpointing
        self.checkpoint_dir = config['artifacts']['save_dir'] + '/checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_losses': [],
            'train_accuracies': [],
            'epochs': []
        }
        
        print("   Loss function: CrossEntropyLoss")
        logging.info("Loss function: CrossEntropyLoss")
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        print(f"   Optimizer: Adam (lr={self.learning_rate})")
        logging.info(f"Optimizer: Adam (lr={self.learning_rate})")
        
        # Initialize best metrics for checkpointing
        self.best_accuracy = 0.0
        
        print("✅ Trainer initialized successfully!")
        logging.info("Trainer initialized successfully!")
    
    def train_one_epoch(self, train_loader, epoch):
        """
        Train the model for one epoch
        
        Args:
            train_loader: DataLoader with training data
            epoch (int): Current epoch number
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        print(f"🔄 Training epoch {epoch + 1}/{self.max_epochs}...")
        logging.info(f"Training epoch {epoch + 1}/{self.max_epochs}...")
        
        # Set model to training mode
        # This enables dropout and batch normalization training behavior
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = len(train_loader)
        
        start_time = time.time()
        
        # Training loop - iterate through all batches
        for batch_idx, (data, target) in enumerate(train_loader):
            
            # Move data to device (CPU or GPU)
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Clear gradients from previous iteration
            # PyTorch accumulates gradients, so we need to clear them
            self.optimizer.zero_grad()
            
            # Forward pass - compute predictions
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass - compute gradients
            loss.backward()
            
            # Update model parameters using gradients
            self.optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            
            # Get predictions for accuracy calculation
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                progress = 100.0 * batch_idx / num_batches
                print(f"   Batch {batch_idx:4d}/{num_batches} ({progress:5.1f}%) - Loss: {loss.item():.4f}")
        
        # Calculate epoch statistics
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_predictions)
        epoch_time = time.time() - start_time
        
        print(f"✅ Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        logging.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        logging.info(f"Average Loss: {avg_loss:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return avg_loss, accuracy
    
    def validate(self, test_loader):
        """
        Validate the model on test data
        
        Args:
            test_loader: DataLoader with test data
            
        Returns:
            tuple: (loss, accuracy, f1_score)
        """
        print("🔍 Validating model...")
        
        # Set model to evaluation mode
        # This disables dropout and sets batch normalization to eval mode
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Disable gradient computation for faster inference
        with torch.no_grad():
            for data, target in test_loader:
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"📊 Validation Results:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   F1-Score: {f1:.4f}")
        
        return avg_loss, accuracy, f1
    
    def train(self, train_loader, test_loader):
        """
        Complete training loop
        
        Args:
            train_loader: DataLoader with training data
            test_loader: DataLoader with test data
            
        Returns:
            dict: Training history and final metrics
        """
        print("Starting training for {} epochs...".format(self.max_epochs))
        print("=" * 60)
        logging.info(f"Starting training for {self.max_epochs} epochs...")
        
        start_time = time.time()
        final_accuracy = 0.0
        
        for epoch in range(self.max_epochs):
            # Train for one epoch
            train_loss, train_accuracy = self.train_one_epoch(train_loader, epoch)
            
            # Store training history
            self.history['train_losses'].append(train_loss)
            self.history['train_accuracies'].append(train_accuracy)
            
            # Validate every epoch
            val_loss, val_accuracy, val_f1 = self.validate(test_loader)
            final_accuracy = val_accuracy
            
            print("-" * 60)
        
        # Save the final model
        self.save_checkpoint(self.max_epochs-1, final_accuracy, is_final=True)
        print("Final model saved")
        logging.info("Final model saved")
        
        total_time = time.time() - start_time
        print("Training completed in {:.1f}s ({:.1f} minutes)".format(total_time, total_time/60))
        print("Final accuracy achieved: {:.4f} ({:.2f}%)".format(final_accuracy, final_accuracy*100))
        
        logging.info(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logging.info(f"Final accuracy achieved: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        # Create training history
        history = {
            'train_losses': self.history['train_losses'],
            'train_accuracies': self.history['train_accuracies'],
            'final_accuracy': final_accuracy,
            'total_time': total_time,
            'epochs_trained': self.max_epochs
        }
        
        return history
    
    def save_checkpoint(self, epoch, accuracy, is_final=False):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch
            accuracy (float): Current accuracy
            is_final (bool): Whether this is the final model
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.config['artifacts']['save_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint data
        # Persist minimal metadata to support generic inference
        data_cfg = self.config.get('data', {})
        model_cfg = self.config.get('model', {})
        normalization = data_cfg.get('normalization', {"mean": [0.1307], "std": [0.3081]})
        # Prefer explicit input_shape; fallback to channels + MNIST size
        input_shape = model_cfg.get('input_shape', [model_cfg.get('input_channels', 1), 28, 28])
        num_classes_meta = model_cfg.get('num_classes', 10)
        model_type = model_cfg.get('type', 'SimpleCNN')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config,
            'metadata': {
                'input_shape': input_shape,
                'normalization': normalization,
                'num_classes': num_classes_meta,
                'model_type': model_type
            }
        }
        
        # Get model name from environment variable, default to original name
        model_name = os.environ.get('MODEL_NAME')
        
        # Save checkpoint
        if is_final:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}.pt')
            torch.save(checkpoint, checkpoint_path)
            print("Final model saved to {}".format(checkpoint_path))
        
        # Always save latest
        latest_path = os.path.join(checkpoint_dir, f'{model_name}-latest.pt')
        torch.save(checkpoint, latest_path)
    
    def plot_training_history(self):
        """
        Plot training history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_losses'], 'b-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_accuracies'], 'r-', label='Training Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.config['artifacts']['save_dir'], 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training history plot saved to {plot_path}")


if __name__ == "__main__":
    print("🧪 This module contains the training functionality.")
    print("   Import it and use the SimpleTrainer class to train models.")
    print("   Example: trainer = SimpleTrainer(model, config)")
    print("            trainer.train(train_loader, test_loader)")
