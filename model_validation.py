import torch
import os
import config
from datasets import create_validation_loader


# Get validation loader
validation_loader = create_validation_loader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = config.RESNET50_MODEL_PATH
if not os.path.exists(model_path):
    raise FileNotFoundError("Can not find model {}".format(model_path))
net = torch.load(model_path).to(device)

net.eval()

total_accuracy = 0
total_samples = 0
with torch.no_grad():
    label_classes = ("healthy", "powdery", "rust")

    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        _, predicted_class = torch.max(outputs, 1)
        for i in range(len(images)):
            predicted_label = label_classes[predicted_class[i].item()]
            true_label = label_classes[labels[i].item()]
            if predicted_label == true_label:
                total_accuracy += 1
            print(f"Predicted label: {predicted_label}, True label: {true_label}")
            total_samples += 1

print(f"Total accuracy: {total_accuracy / total_samples:.2f}")
