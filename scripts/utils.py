import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
    transforms.Grayscale(),  
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Chuẩn hóa theo MNIST
])


    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
