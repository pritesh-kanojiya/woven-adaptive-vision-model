"""
Simple CNN Model for MNIST Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - Input: 28x28 grayscale images (1 channel)
    - Conv Layer 1: 32 filters, 3x3 kernel + ReLU + MaxPool
    - Conv Layer 2: 64 filters, 3x3 kernel + ReLU + MaxPool  
    - Flatten + Dropout
    - Fully Connected 1: 128 neurons + ReLU + Dropout
    - Fully Connected 2: 10 outputs (one for each digit)
    
    Expected accuracy: 95%+ on MNIST
    """
    
    def __init__(self, input_channels=1, num_classes=10):
        """
        Initialize the CNN layers
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale MNIST)
            num_classes (int): Number of output classes (10 for digits 0-9)
        """
        super(SimpleCNN, self).__init__()
        
        print("🏗️  Building SimpleCNN model...")
        print(f"   Input channels: {input_channels}")
        print(f"   Output classes: {num_classes}")
        
        # First convolutional block
        # Input: 1x28x28 → Output: 32x26x26 (28-3+1=26)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,   # 1 channel (grayscale)
            out_channels=32,              # 32 feature maps
            kernel_size=3,                # 3x3 filter
            padding=0                     # no padding
        )
        
        # Second convolutional block  
        # Input: 32x13x13 → Output: 64x11x11 (13-3+1=11)
        self.conv2 = nn.Conv2d(
            in_channels=32,               # from first conv layer
            out_channels=64,              # 64 feature maps
            kernel_size=3,                # 3x3 filter
            padding=0                     # no padding
        )
        
        # Max pooling layer - reduces spatial dimensions by half
        # Applied after each conv layer: 26→13, 11→5
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layers - prevent overfitting by randomly setting neurons to 0
        self.dropout1 = nn.Dropout(0.25)  # 25% dropout after conv layers
        self.dropout2 = nn.Dropout(0.5)   # 50% dropout after first FC layer
        
        # Fully connected layers
        # After conv1+pool: 32x13x13, after conv2+pool: 64x5x5 = 1600 features
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Hidden layer with 128 neurons
        self.fc2 = nn.Linear(128, num_classes)  # Output layer (10 classes)
        
        print("✅ Model created successfully!")
        print(f"   Total parameters: {self.count_parameters():,}")
    
    def forward(self, x):
        """
        Forward pass - how data flows through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, 10) - logits for each digit
        """
        
        # First convolutional block
        # x: (batch, 1, 28, 28) → (batch, 32, 26, 26)
        x = self.conv1(x)
        x = F.relu(x)           # Apply ReLU activation (adds non-linearity)
        x = self.pool(x)        # Apply max pooling: (batch, 32, 26, 26) → (batch, 32, 13, 13)
        
        # Second convolutional block  
        # x: (batch, 32, 13, 13) → (batch, 64, 11, 11)
        x = self.conv2(x)
        x = F.relu(x)           # Apply ReLU activation
        x = self.pool(x)        # Apply max pooling: (batch, 64, 11, 11) → (batch, 64, 5, 5)
        
        # Flatten for fully connected layers
        # x: (batch, 64, 5, 5) → (batch, 1600)
        x = torch.flatten(x, 1)  # Keep batch dimension, flatten everything else
        x = self.dropout1(x)     # Apply dropout for regularization
        
        # First fully connected layer
        # x: (batch, 1600) → (batch, 128)
        x = self.fc1(x)
        x = F.relu(x)           # Apply ReLU activation
        x = self.dropout2(x)    # Apply dropout
        
        # Output layer - no activation (will be handled by loss function)
        # x: (batch, 128) → (batch, 10)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        """
        Print a summary of the model architecture
        """
        print("\n🏗️  Model Architecture Summary:")
        print("=" * 50)
        print("Layer               | Output Shape    | Parameters")
        print("-" * 50)
        print("Input               | (1, 28, 28)     | 0")
        print("Conv2d + ReLU       | (32, 26, 26)    | 320")
        print("MaxPool2d           | (32, 13, 13)    | 0") 
        print("Conv2d + ReLU       | (64, 11, 11)    | 18,496")
        print("MaxPool2d           | (64, 5, 5)      | 0")
        print("Flatten + Dropout   | (1600,)         | 0")
        print("Linear + ReLU       | (128,)          | 204,928")
        print("Dropout             | (128,)          | 0")
        print("Linear (output)     | (10,)           | 1,290")
        print("-" * 50)
        print(f"Total Parameters: {self.count_parameters():,}")
        print("=" * 50)


def create_model(config):
    """
    Factory function to create a model based on configuration
    
    Args:
        config (dict): Configuration dictionary containing model settings
        
    Returns:
        SimpleCNN: A new instance of the CNN model
    """
    print("🚀 Creating model from configuration...")
    
    # Extract model configuration
    model_config = config['model']
    input_channels = model_config.get('input_channels', 1)
    num_classes = model_config.get('num_classes', 10)
    
    # Create the model
    model = SimpleCNN(input_channels=input_channels, num_classes=num_classes)
    
    # Print architecture summary
    model.print_architecture()
    
    return model


def test_model():
    """
    Simple test function to verify the model works correctly
    """
    print("🧪 Testing model with dummy data...")
    
    # Create a dummy configuration
    config = {
        'model': {
            'input_channels': 1,
            'num_classes': 10
        }
    }
    
    # Create model
    model = create_model(config)
    
    # Create dummy input (batch of 4 MNIST images)
    dummy_input = torch.randn(4, 1, 28, 28)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():  # No gradients needed for testing
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (first sample): {output[0]}")
    
    # Check if output makes sense
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"
    print("✅ Model test passed!")


if __name__ == "__main__":
    # Run test when file is executed directly
    test_model()
