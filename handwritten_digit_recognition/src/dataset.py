from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64):
    root_path = './data/MNIST'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    training_data = datasets.MNIST(root=root_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=root_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
