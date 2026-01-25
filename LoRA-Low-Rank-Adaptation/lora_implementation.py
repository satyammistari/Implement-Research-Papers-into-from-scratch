"""
LoRA: Low-Rank Adaptation Implementation from Scratch
Paper: https://arxiv.org/abs/2106.09685

This script demonstrates a complete LoRA implementation using PyTorch,
including training on MNIST and fine-tuning with minimal parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F


# ============================================================================
# 1. LoRA LAYER IMPLEMENTATION
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer
    
    Wraps a Linear layer and adds trainable low-rank matrices A and B.
    The output is: original_layer(x) + (x @ B.T @ A.T) * scaling
    
    Args:
        original_layer: The Linear layer to adapt
        r: Rank of the low-rank matrices (smaller = fewer parameters)
        alpha: Scaling factor (typically same as r)
    """
    def __init__(self, original_layer, r=4, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer

        # Freeze original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Get dimensions
        features_in = original_layer.in_features
        features_out = original_layer.out_features
        
        # Expose these as attributes so they can be accessed
        self.in_features = features_in
        self.out_features = features_out

        # LoRA matrices - initialized with random gaussian noise
        self.lora_A = nn.Parameter(torch.randn(features_out, r))
        self.lora_B = nn.Parameter(torch.randn(r, features_in))
        self.scaling = alpha / r

    def forward(self, x):
        # Original frozen path
        original_output = self.original_layer(x)
        
        # LoRA path - apply B then A to the input
        lora_output = (x @ self.lora_B.T @ self.lora_A.T) * self.scaling

        return original_output + lora_output


# ============================================================================
# 2. BASELINE MODEL
# ============================================================================

class RichBoyNet(nn.Module):
    """A simple feedforward network for MNIST classification"""
    def __init__(self, hidden_size1=1000, hidden_size2=500, num_classes=10):
        super(RichBoyNet, self).__init__()
        
        self.linear1 = nn.Linear(28*28, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, 20)
        self.relu = nn.ReLU()
        
    def forward(self, imgs):
        x = imgs.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# ============================================================================
# 3. TRAINING UTILITIES
# ============================================================================

def train(model, loader, epochs=1, device='cpu'):
    """Generic training function"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})


def count_trainable_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    # Set device
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("\nDownloading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                  transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Step 1: Pre-train the model
    print("\n" + "="*70)
    print("STEP 1: Pre-training full model on MNIST")
    print("="*70)
    
    model = RichBoyNet().to(device)
    train(model, train_loader, epochs=5, device=device)

    # Step 2: Inject LoRA layers
    print("\n" + "="*70)
    print("STEP 2: Injecting LoRA layers")
    print("="*70)

    # Freeze the entire network first
    for param in model.parameters():
        param.requires_grad = False

    original_param_count = sum(p.numel() for p in model.parameters())
    print(f"Original Parameters (Frozen): {original_param_count:,}")

    # Inject LoRA (only these new parameters will be trainable)
    model.linear2 = LoRALayer(model.linear2, r=16, alpha=16).to(device)
    model.linear3 = LoRALayer(model.linear3, r=16, alpha=16).to(device)

    new_trainable_params = count_trainable_parameters(model)
    print(f"New Trainable Parameters (LoRA only): {new_trainable_params:,}")
    
    efficiency = new_trainable_params / original_param_count * 100
    print(f"Parameter Efficiency: {efficiency:.3f}% of original size")

    # Step 3: Fine-tune on digit 9
    print("\n" + "="*70)
    print("STEP 3: Fine-tuning ONLY LoRA parameters on digit 9")
    print("="*70)

    # Create dataset with only digit 9
    digit_9_indices = train_dataset.targets == 9
    train_dataset_9 = torch.utils.data.Subset(
        train_dataset, torch.where(digit_9_indices)[0]
    )
    loader_9 = DataLoader(train_dataset_9, batch_size=10, shuffle=True)

    train(model, loader_9, epochs=1, device=device)

    # Step 4: Evaluate
    print("\n" + "="*70)
    print("STEP 4: Evaluating Results")
    print("="*70)

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            
            # Filter to only check accuracy on digit 9
            mask = (y == 9)
            if mask.sum() > 0:
                correct += (predicted[mask] == y[mask]).sum().item()
                total += mask.sum().item()

    accuracy = 100 * correct / total
    print(f"\nAccuracy on Digit '9' after LoRA fine-tuning: {accuracy:.2f}%")
    print("\n✅ Implementation Complete!")
    
    return model


if __name__ == "__main__":
    model = main()
