import torch
import torch.nn as nn
from scripts.utils import get_data_loaders
from models.cnn import CNN
from sklearn.metrics import accuracy_score

def evaluate_model(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _, test_loader = get_data_loaders()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            print("Output model:", outputs[:5])  # In ra 5 output đầu tiên
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Độ chính xác trên tập test: {acc:.4f}")

if __name__ == "__main__":
    print("Testing CNN...")
    evaluate_model(CNN())
