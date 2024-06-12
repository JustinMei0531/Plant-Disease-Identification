import torch
from tqdm import tqdm
from datasets import create_test_loader
# Load test loader
test_loader = create_test_loader()
# Get runtime device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model state
net = torch.load("./models/resnet34/resnet34-plant-disease-recognition.pt")

# Set model to test mode
net.eval()

correct = 0
total = 0

with torch.no_grad():
    for index, (images, labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")
