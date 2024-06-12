import torch
from dataset import get_data_loaders
from model import LeNet


def evaluate_model(batch_size=64):
    _, test_loader = get_data_loaders(batch_size)
    model = LeNet()
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')
