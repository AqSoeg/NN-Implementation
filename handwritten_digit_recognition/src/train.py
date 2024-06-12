import torch
import torch.optim as optim
import torch.nn as nn
from model import LeNet
from dataset import get_data_loaders
from torch.utils.tensorboard import SummaryWriter


def train_model(epochs=10, batch_size=64, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model training using device {device}')

    writer = SummaryWriter()

    train_loader, _ = get_data_loaders(batch_size)
    model = LeNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), epoch)

            if batch_idx % 100 == 0:
                print(f'Train epoch {epoch + 1}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), './model.pth')
