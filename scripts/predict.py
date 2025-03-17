import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import matplotlib
matplotlib.use('Agg')  # Fix lỗi hiển thị trên WSL
import matplotlib.pyplot as plt
from models.cnn import CNN

# Load model đã huấn luyện
model = CNN()
model.load_state_dict(torch.load("models/cnn.pth"))
model.eval()

# Transform ảnh giống như trong quá trình huấn luyện
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),  # Resize ảnh về 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Đọc ảnh từ đường dẫn
def predict_image(image_path):
    image = Image.open(image_path).convert("L").resize((28, 28))  # Chuyển về grayscale & resize
    print(f"Kích thước ảnh: {image.size}")  # Kiểm tra kích thước ảnh

    plt.imshow(image, cmap="gray")
    plt.title("Ảnh đầu vào")
    plt.savefig("image_preview.png")  # Lưu ảnh kiểm tra

    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    print(f"Kích thước tensor đầu vào: {image.shape}")  # Kiểm tra tensor đầu vào

    with torch.no_grad():
        output = model(image)
        print("Output model:", output)  # In toàn bộ output tensor
        _, predicted = torch.max(output, 1)

    print(f"Số dự đoán: {predicted.item()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py path_to_image")
    else:
        predict_image(sys.argv[1])
