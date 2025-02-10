# Import necessary libraries
import torch  
import torch.nn as nn  
import torch.optim as optiim  
from torchvision import transforms 
import torchvision.models as models  
from torch.utils.data import DataLoader, Dataset,TensorDataset 
from medmnist import BreastMNIST, BloodMNIST  
import matplotlib.pyplot as plt  
import numpy as np  
import os

# Set random seed for reproducibility
torch.manual_seed(42)

#Prepares and returns data loaders for training, validation and test sets from medmnist api
def load_dataloader(batch_size=32):
    # Define image transformations pipeline for RGB images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # RGB normalization
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip
        transforms.RandomRotation(degrees=(-10, 10)), # Random rotation for robustness
    ])

    # Load BloodMNIST dataset splits with transformations
    train_dataset = BloodMNIST(split='train', transform=transform, download=True)
    val_dataset = BloodMNIST(split='val', transform=transform, download=True)
    test_dataset = BloodMNIST(split='test', transform=transform, download=True)
    
    # Create data loaders with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle training data
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)     # No need to shuffle val/test
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
'''
#Prepares and returns data loaders for training, validation and test sets from .npz file and medmnist api respectively
def load_dataloader(batch_size=32):
    # Define image transformations pipeline
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # RGB normalization
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
        transforms.RandomRotation(degrees=(-10, 10)),  # Random rotation between -10 and 10 degrees
    ])

    # Load the .npz file for training data
    data = np.load('AMLS-Coursework/Datasets/BloodMNIST/bloodmnist.npz')
    
    # Extract training data and labels
    train_images = data['train_images']
    train_labels = data['train_labels']
    
    # Calculate split sizes for training data
    total_size = len(train_images)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Split training data into train and validation sets
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train and validation sets
    train_images_split = train_images[train_indices]
    train_labels_split = train_labels[train_indices]
    val_images = train_images[val_indices]
    val_labels = train_labels[val_indices]
    
    # Convert numpy arrays to torch tensors and add channel dimension
    train_images_tensor = torch.FloatTensor(train_images_split).unsqueeze(1) / 255.0  # Normalize to [0,1]
    train_labels_tensor = torch.FloatTensor(train_labels_split)
    val_images_tensor = torch.FloatTensor(val_images).unsqueeze(1) / 255.0
    val_labels_tensor = torch.FloatTensor(val_labels)
    
    # Apply transformations to training and validation data
   # Convert numpy array to a tensor
    train_images_tensor = torch.FloatTensor(train_images_split) / 255.0  # shape: (N, H, W, 3)

    # Permute the dimensions to get (N, 3, H, W)
    train_images_tensor = train_images_tensor.permute(0, 3, 1, 2)

    val_images_tensor = torch.FloatTensor(val_images) / 255.0  # shape: (N, H, W, 3)
    val_images_tensor = val_images_tensor.permute(0, 3, 1, 2)
    
    # Create training and validation datasets
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
    
    # Load test set from MedMNIST
    test_dataset = BloodMNIST(split='test', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
        transforms.RandomRotation(degrees=(-10, 10)),  # Random rotation between -10 and 10 degrees
    ]), download=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader
'''
class BloodCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(BloodCNN, self).__init__()
        # Feature extraction layers
        self.feats = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 28x28x3 -> Output: 28x28x32
            nn.BatchNorm2d(32),  # Normalize activations for stable training
            nn.ReLU(inplace=True),  # Non-linear activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Spatial reduction: 14x14x32
            nn.Dropout(0.25),  # Prevent overfitting
            
            # Second convolutional block - Similar structure with more filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 7x7x64
            nn.Dropout(0.25),
            
            # Third convolutional block - Deepest features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 7x7x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 3x3x128
            nn.Dropout(0.25),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten 3D feature maps to 1D vector
            nn.Linear(3*3*128, 512),  # Fully connected layer
            nn.BatchNorm1d(512),  # Normalize activations
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Higher dropout in classifier
            nn.Linear(512, num_classes)  # Output layer (8 classes)
        )
    
    def forward(self, x):
        x = self.feats(x)  # Extract features
        return self.classifier(x)  # Classify features

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        # Move data to device and prepare targets
        inputs = inputs.to(device)
        targets = targets.reshape(-1).to(device).long()  # Reshape to 1D for CrossEntropyLoss
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    return running_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 8
    class_total = [0] * 8
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze().long()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Print per-class accuracy
    for i in range(8):
        if class_total[i] > 0:
            print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return running_loss / len(val_loader), correct / total

def plot_trainning_history(train_losses, train_accs, val_losses, val_accs, save_path):
   
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

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training hyperparameters
    LEARNING_RATE = 0.001
    N_EPOCHS = 50
    BATCH_SIZE = 16
    
    print("Loading data....")
    train_loader, val_loader, test_loader = load_dataloader(batch_size=BATCH_SIZE)
    
    print("Initialising model...")
    model = BloodCNN(num_classes=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optiim.Adam(model.parameters(), 
                           lr=LEARNING_RATE, 
                           weight_decay=1e-4)  # L2 regularization
    
    #scheduler = optiim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='max', factor=0.1, patience=5, verbose=True
    #)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    
    # Create path for saving model
    model_dir = 'B'  
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'best_model.pth')
    plot_path = os.path.join(model_dir, 'training_history.png')
    
    print("Starting training....")
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        #scheduler.step(val_acc)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch + 1}/{N_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")     
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")   
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
    
    plot_trainning_history(train_losses, train_accs, val_losses, val_accs, plot_path)
    
    print("\nEvaluating best model on test dataset...")
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
if __name__ == '__main__':
    main()