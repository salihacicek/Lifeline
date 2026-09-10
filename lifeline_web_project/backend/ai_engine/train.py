import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import HybridECGModel, FocalLoss
from dataset import augment_ecg_data

def train_model(epochs=2):
    """
    Training loop for the 1D-CNN + BiLSTM model.
    Using 'mps' for Apple Silicon GPU acceleration.
    """
    # 1. Hardware Detection (MPS for Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon GPU (MPS) detected! Training will be accelerated.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("NVIDIA GPU (CUDA) detected.")
    else:
        device = torch.device("cpu")
        print("No GPU detected, falling back to CPU.")

    # 2. Simulated Dataset (1000 samples length, 1 channel)
    print("Generating simulated data for testing...")
    num_samples = 100
    seq_length = 1000
    
    X_dummy = np.random.randn(num_samples, seq_length)
    y_dummy = np.random.randint(0, 2, num_samples) # 0 or 1
    
    # 3. Data Augmentation
    X_aug, y_aug = augment_ecg_data(X_dummy, y_dummy)
    
    # Convert to PyTorch Tensors: (Batch, Channels, Length)
    X_tensor = torch.tensor(X_aug, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_aug, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 4. Model Initialization
    model = HybridECGModel(input_channels=1, num_classes=1).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training Loop
    print(f"Starting training for {epochs} epochs on {device}...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")
        
    print("Training Complete! Saving model...")
    torch.save(model.state_dict(), "ecg_model.pth")
    print("Model saved to 'ecg_model.pth'.")

if __name__ == "__main__":
    # Test the training loop with just 2 epochs locally so it doesn't freeze the Mac
    train_model(epochs=2)
