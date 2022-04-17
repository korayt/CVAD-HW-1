import torch
import torchvision
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.cilrs import CILRS
import matplotlib.pyplot as plt
import numpy as np
import PIL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    for i, (images, measurements) in enumerate(dataloader):
        command, speed, throttle, brake, steer,_,_,_,_ = measurements
        images = images/255
        images = images.to(device)
        command = command.to(device)
        speed = speed.to(device)
        throttle = throttle.to(device)
        brake = brake.to(device)
        steer = steer.to(device)
        # Forward pass
        outputs = model(images, command, speed)
        criterion = torch.nn.L1Loss()
        loss1 = criterion(outputs[0], throttle)
        loss2 = criterion(outputs[1], brake)
        loss3 = criterion(outputs[2], steer)
        loss = loss1 + loss2 + loss3
        return loss.item()


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    learning_rate = 0.001
    for i, (images, measurements) in enumerate(dataloader):
        command, speed, throttle, brake, steer,_,_,_,_ = measurements
        images = images/255
        images = images.to(device)
        command = command.to(device)
        speed = speed.to(device)
        throttle = throttle.to(device)
        brake = brake.to(device)
        steer = steer.to(device)

        # Forward pass
        outputs = model(images, command, speed)
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print('OUTPUTS THROTTLE BRAKE STEER')
        print(outputs)
        print('throttle')
        print(throttle)
        print('brake')
        print(brake)
        print('steer')
        print(steer)
        loss1 = criterion(outputs[0], throttle)
        loss2 = criterion(outputs[1], brake)
        loss3 = criterion(outputs[2], steer)
        loss = loss1 + loss2 + loss3
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    y = [i+1 for i in range(10)]
    
    # plot lines
    plt.plot(y, train_loss, label = "training loss")
    plt.plot(y, val_loss, label = "validation loss")
    plt.legend()
    plt.show()

def main():
    # Change these paths to the correct paths in your downloaded expert dataset

    train_root = '/home/koray/expert_data/train'
    val_root = '/home/koray/expert_data/val'
    model = CILRS()
    train_dataset = ExpertDataset(train_root, 2761)
    val_dataset = ExpertDataset(val_root, 300)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_loss = train(model, train_loader)
        train_losses.append(train_loss)
        val_loss = validate(model, val_loader)
        val_losses.append(val_loss)
        print(f'Epoch [{i + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
