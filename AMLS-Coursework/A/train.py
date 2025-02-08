# Import necessary libraries
import torch  
import torch.nn as nn  
import torch.optim as optiim  
from torchvision import transforms 
import torchvision.models as models  
from torch.utils.data import DataLoader, Dataset 
from medmnist import BreastMNIST, BloodMNIST  
import matplotlib.pyplot as plt  
import numpy as np  
import os

# Set random seed for reproducibility
torch.manual_seed(42)

#Prepares and returns data loaders for training, validation and test sets
def load_dataloader(batch_size=32):
    # Define image transformations pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values to [-1, 1]
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
        transforms.RandomRotation(degrees=(-10, 10)),  # Random rotation between -10 and 10 degrees
    ])

    # Load datasets with the defined transformations
    train_dataset = BreastMNIST(split='train', transform=transform, download=True)
    val_dataset = BreastMNIST(split='val', transform=transform, download=True)
    test_dataset = BreastMNIST(split='test', transform=transform, download=True)

    # Create data loaders
    # Training loader shuffles data to prevent learning order-specific patterns
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Validation and test loaders don't need shuffling
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class BreastCNN(nn.Module):    #Custom CNN architecture for breast cancer classification
   
    def __init__(self):
        super(BreastCNN, self).__init__()
        # Feature extraction layers
        self.feats = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: 1 channel (grayscale), Output: 32 channels
            nn.BatchNorm2d(32),  # Batch normalization for faster convergence
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions
            nn.Dropout(0.25),  # Prevent overfitting
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the 3D feature maps to 1D
            nn.Linear(128 * 3 * 3, 128),  # Fully connected layer
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Output layer
            nn.Sigmoid()  # For binary classification
        )
    
    def forward(self, x):#forward pass through the network
        return self.classifier(self.feats(x))
    

    
def train_epoch(model, train_loader, optimizer, criterion, device):#Trains the model for one epoch
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate over batches of data
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device).float()# Move data to specified device (CPU/GPU)
        optimizer.zero_grad()  # Zero the parameter gradients to prevent accumulation
        outputs = model(inputs)# Forward pass: compute model predictions
        loss = criterion(outputs, targets)# Calculate loss
        loss.backward()# Backward pass: compute gradient of the loss with respect to model parameters
        
        # Update model parameters
        optimizer.step()
        
        # Accumulate statistics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
        total += targets.size(0)  # Count total number of samples
        correct += (predicted == targets).sum().item()  # Count correct predictions
        
    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device): #Evaluates the model on validation/test data
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
    # Calculate average loss and accuracy
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc

def plot_trainning_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """
    Plots training and validation metrics over time
    
    Args:
        train_losses (list): Training losses for each epoch
        train_accs (list): Training accuracies for each epoch
        val_losses (list): Validation losses for each epoch
        val_accs (list): Validation accuracies for each epoch
        save_path (str): Path where to save the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training and validation losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save plot to specified path
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():#Main training pipeline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training hyperparameters
    LEARNING_RATE = 0.0001
    N_EPOCHS = 10
    BATCH_SIZE = 16
    
    # Load and prepare data
    print("Loading data....")
    train_loader, val_loader, test_loader = load_dataloader(batch_size=BATCH_SIZE)
    
    # Initialize model
    print("Initialising model...")
    model = BreastCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optiim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Initialize tracking variables
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0

    # Create path for saving model
    model_dir = 'A'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'best_model.pth')
    plot_path = os.path.join(model_dir, 'training_history.png')

    # Training loop
    print("Starting training....")
    for epoch in range(N_EPOCHS):
        # Train one epoch and get metrics
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        # Evaluate on validation set
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch + 1} / {N_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")     
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")   
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)  # Save model in A folder
            
    # Plot training history and save in A folder
    plot_trainning_history(train_losses, train_accs, val_losses, val_accs, plot_path)
    
    # Final evaluation on test set
    print("\nEvaluating best model on test dataset...")
    model.load_state_dict(torch.load(model_path))  # Load model from A folder
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
if __name__ == '__main__':
    main()