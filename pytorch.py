import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import lab_to_rgb, imshow
from data_extraction_test import shapeX, shapeY, test_image_bnw, entrees, sorties_a, sorties_b, test_image_l

import torch.nn.functional as F

# Define the neural network class (correcting Conv2d and padding issues)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        layer_size = 64
        kernel_shape = 4

        # The output size will change with stride and padding, so adjusting layers accordingly
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),  # Adjust padding
            nn.ReLU(),
            nn.Conv2d(in_channels=layer_size, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=layer_size, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=layer_size, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=layer_size, out_channels=layer_size, kernel_size=kernel_shape, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=layer_size, out_channels=1, kernel_size=kernel_shape, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Set the device to DirectML for
device = torch_directml.device()

if device == torch_directml.device():
    print("Model is on : GPU")
else:
    print("Model is on : CPU")


# Initialize the models
model_a = SimpleModel().to(device)
model_b = SimpleModel().to(device)

def load_models():
    model_a.load_state_dict(torch.load("model_a_weights.pth"))
    model_b.load_state_dict(torch.load("model_b_weights.pth"))

def train(n=1):
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
    optimizer_b = optim.Adam(model_b.parameters(), lr=0.001)
    
    # Convert your data to PyTorch tensors (corrected dtype)
    entrees_tensor = torch.tensor(entrees, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    sorties_a_tensor = torch.tensor(sorties_a, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    sorties_b_tensor = torch.tensor(sorties_b, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension

    # Create DataLoader for batches (optional)
    train_data_a = TensorDataset(entrees_tensor, sorties_a_tensor)
    train_data_b = TensorDataset(entrees_tensor, sorties_b_tensor)
    train_loader_a = DataLoader(train_data_a, batch_size=16, shuffle=True)
    train_loader_b = DataLoader(train_data_b, batch_size=16, shuffle=True)
    
    model_a.train()
    model_b.train()
    
    for epoch in range(n):
        for i, (inputs, targets_a) in enumerate(train_loader_a):
            # Zero the gradients
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()

            # Forward pass for model A
            outputs_a = model_a(inputs)
            outputs_a_resized = F.interpolate(outputs_a, size=targets_a.shape[2:], mode='bilinear', align_corners=False)  # Resize

            # Compute loss for model A
            loss_a = criterion(outputs_a_resized, targets_a)

            # Backward pass and optimization for model A
            loss_a.backward()
            optimizer_a.step()

            # Forward pass for model B
            outputs_b = model_b(inputs)
            outputs_b_resized = F.interpolate(outputs_b, size=sorties_b_tensor.shape[2:], mode='bilinear', align_corners=False)  # Resize

            # Compute loss for model B
            loss_b = criterion(outputs_b_resized, sorties_b_tensor[i])  # Use appropriate target for model B

            # Backward pass and optimization for model B
            loss_b.backward()
            optimizer_b.step()

            # Print loss every few batches
            if i == 1 or (i>0 and i % 100 == 0) :
                print(f"Epoch {epoch+1}/{n}, Batch {i}, Loss A: {loss_a.item():.4f}, Loss B: {loss_b.item():.4f}")
    # test()
    
def test():
    # After training, run the models for prediction
    model_a.eval()  # Set the model to evaluation mode
    model_b.eval()  # Set the model to evaluation mode
    
    # Ensure that the input is converted into a tensor and has the correct shape
    test_image_bnw_tensor = torch.tensor(np.array([test_image_bnw]), dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension

    # Predict using the models
    new_test_image_a = model_a(test_image_bnw_tensor).detach().cpu().numpy()[0]  # Get the prediction for model A
    new_test_image_b = model_b(test_image_bnw_tensor).detach().cpu().numpy()[0]  # Get the prediction for model B

    
    # Remove the channel dimension from model outputs
    new_test_image_a = new_test_image_a.squeeze()  # Remove extra channel dimension, shape should be (32, 32)
    new_test_image_b = new_test_image_b.squeeze()  # Remove extra channel dimension, shape should be (32, 32)
    
    # Now concatenate the images along the last axis (axis=-1)
    new_test_image_lab = np.concatenate((
        test_image_l[:,:,0][..., np.newaxis],  # Add back the channel dimension for proper concatenation
        new_test_image_a[..., np.newaxis],  # Add back the channel dimension
        new_test_image_b[..., np.newaxis],  # Add back the channel dimension
    ), axis=-1)
    
    
    # Adjust the LAB values
    new_test_image_lab[:,:,1] -= 128
    new_test_image_lab[:,:,2] -= 128
    new_test_image_lab[:,:,0] /= 2.55
    
    # Show the result as RGB
    imshow(lab_to_rgb(new_test_image_lab), False)
