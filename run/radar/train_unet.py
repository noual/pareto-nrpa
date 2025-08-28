import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
from datetime import datetime

sys.path.append('../../')
sys.path.append('..')
from search_spaces.radar.radar_dataset import RadarDavaDataset
from search_spaces.radar.radar_node import NASBench201UNet
from search_spaces.radar.unet import UNet
from tqdm import tqdm

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_unet(data_path, cell_str, epochs=200, batch_size=16, learning_rate=1e-5, name="model", in_features=16):
    features = [in_features, in_features*2, in_features*4, in_features*8]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloaders
    dataset = RadarDavaDataset(root_dir=data_path, batch_size=batch_size, has_distance=True)
    train_loader, val_loader, test_loader = dataset.generate_loaders()

    # Initialize model, loss and optimizer
    if cell_str == "unet":
        model = UNet(in_channels=1, features=features).to(device)
    else:
        model = NASBench201UNet(cell_str= '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_1x1~2|+|nor_conv_3x3~0|skip_connect~1|avg_pool_3x3~2|skip_connect~3|+|avg_pool_3x3~0|skip_connect~1|nor_conv_3x3~2|nor_conv_3x3~3|none~4|',
                                input_size=128, input_depth=1, n_vertices=5, features=features)
        model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=f"runs/{name}")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        k = 0
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            k += 1
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if np.random.randint(30) == 7:
                input_img = inputs[0].detach().cpu()
                outputs = model(inputs.to(device))
                pred_img = torch.sigmoid(outputs[0]).detach().cpu()
                label_img = targets[0].detach().cpu()
                # Add images to TensorBoard
                writer.add_image("Input", input_img, epoch*len(train_loader) + k, dataformats="CHW")
                writer.add_image("Prediction", pred_img, epoch*len(train_loader) + k, dataformats="CHW")
                writer.add_image("Label", label_img, epoch*len(train_loader) + k, dataformats="CHW")
                writer.add_scalar('Loss/train_step', loss.item(), epoch*len(train_loader) + k)


        # Initialize tensorboard writer

        train_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Log training and validation loss
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')

    writer.close()
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('--cell_str', type=str, required=True, help='Cell structure string')
    parser.add_argument('--in_features', type=int, default=16, help='Number of input features')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--name', type=str, default=datetime.now().strftime("%d_%H_%M"), help='Run name')

    args = parser.parse_args()
    DATA_PATH = "../../data/radar/training_set/mat"
    model = train_unet(DATA_PATH, args.cell_str, epochs=args.epochs,
                       learning_rate=args.lr, batch_size=args.batch_size,
                       name=args.name, in_features=args.in_features)
    torch.save(model.state_dict(), f'./runs/{args.name}_net.pth')
    print(
        f"Model saved as ./runs/{args.name}_net.pth"
    )
