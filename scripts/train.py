import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Sử dụng chế độ không cần GUI
import matplotlib.pyplot as plt
from scripts.utils import get_data_loaders
from models.cnn import CNN

def visualize_data():
    """ Kiểm tra ảnh dữ liệu đầu vào """
    train_loader, _ = get_data_loaders(batch_size=5)  # Lấy 5 ảnh để kiểm tra
    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.savefig("train_sample.png")
    print("📸 Đã lưu ảnh mẫu vào train_sample.png")

def train_model(model, epochs=10, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, _ = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 🛠 Đổi optimizer nếu cần

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient {name}: {param.grad.abs().mean().item()}")
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"🛠 Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/cnn.pth")
    print("💾 Đã lưu mô hình tại models/cnn.pth")

if __name__ == "__main__":
    visualize_data()  # Kiểm tra ảnh đầu vào
    print("🚀 Bắt đầu huấn luyện...")
    train_model(CNN(), epochs=20, lr=0.005)  # Giảm learning rate
