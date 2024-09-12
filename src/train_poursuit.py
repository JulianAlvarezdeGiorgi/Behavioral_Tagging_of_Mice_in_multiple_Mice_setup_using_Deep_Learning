import numpy as np
import dataloader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import os
import time
from torch_geometric.nn import GATv2Conv, global_mean_pool
# reload library
import importlib
import cv2
#import utils as ut
import pandas as pd
import DataDLC
from torch_geometric.data import Data, DataLoader
import tqdm
import time
import augmentation

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pickle as pkl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load the data
dataset = torch.load(r'c:\Users\jalvarez\Documents\Data\Dataset_DMDmaleMDX5CVmalefem\dataset_large.pkl', map_location=device)

# Select the behaviour to classify (Sniffing in this case)
indx_behaviour1 = 2
indx_behaviour2 = 7

print('Merging the behaviours')
#augmentation.merge_symetric_behaviours(indx_behaviour1, indx_behaviour2, dataset)

for i in range(len(dataset)):
    dataset[i].behaviour = dataset[i].behaviour[2]
print('Done selecting the behaviour')

# Suffle the dataset
np.random.seed(0)
np.random.shuffle(dataset)

# Split train and test
train_size = int(0.8 * len(dataset))

train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

print('The train dataset has %d samples' % len(train_dataset))
print('The test dataset has %d samples' % len(test_dataset))


batch_size = 32

# Create the dataloaders for train, validation and test
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
graphencoder = models.GATEncoder(nout = 8, nhid=16, attention_hidden=2, n_in=4, dropout=0.5)
class_head = models.ClassificationHead(n_latent=576, nhid = 32, nout = 2)

model = models.GraphClassifier(graphencoder, class_head)

model.to(device)

print('The model has %d trainable parameters' % sum(p.numel() for p in model.parameters() if p.requires_grad))


# Directory to save the model checkpoints
checkpoint_dir = r'/gpfs/users/alvarezj/workdir/Project/Data/Checkpoints/GAT_with_residuals_FullGraph_sniff_R'
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, loss, path):
    # Save the model, optimizer state, epoch, and loss
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

# Training loop
num_epochs = 200
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir='runs/GAT_with_residuals_FullGraph_sniff_R')  # TensorBoard writer

start_time = time.time()  # Time the training
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    i = 0

    for data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        labels = data.behaviour

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Log training loss and accuracy at each step
        if i % 100 == 0:  # Log every 100 iterations, adjust as needed
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Accuracy/Train', correct / total, epoch * len(train_loader) + i)
        i += 1

    train_accuracy = correct / total
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Training Accuracy: {train_accuracy}")

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_data in tqdm.tqdm(test_loader):
            val_outputs = model(val_data)
            val_labels = val_data.behaviour
            val_loss += criterion(val_outputs, val_labels).item()
            val_predicted = val_outputs.argmax(dim=1)
            correct += (val_predicted == val_labels).sum().item()
            total += val_labels.size(0)

    val_accuracy = correct / total
    avg_val_loss = val_loss / len(test_loader)

    print(f"Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")

    # Log validation metrics
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    # Step the scheduler
    #scheduler.step()

    # Log learning rate
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_lr, epoch)
    print(f"Learning Rate after epoch {epoch + 1}: {current_lr}")

    # Save checkpoint after each 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, checkpoint_path)

# Save the final model
if num_epochs % 5 != 0:
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, checkpoint_path)

# Time the training
end_time = time.time()
print(f"Training took {end_time - start_time} seconds, for {num_epochs} epochs")
# Close the TensorBoard writer
writer.close()

print("Training finished")