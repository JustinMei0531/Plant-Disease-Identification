import torch
from tqdm import tqdm
import os
from datasets import create_test_loader
import config

# Load test loader
test_loader = create_test_loader()
# Get runtime device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = config.RESNET50_MODEL_PATH
if not os.path.exists(model_path):
    raise FileNotFoundError("Can not find model {}".format(model_path))
net = torch.load(model_path)

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
