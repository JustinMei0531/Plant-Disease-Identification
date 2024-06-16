import torch
from torch import optim, nn
from tqdm import tqdm
from model import ResNet34, ResNet50, ResNet101
from datasets import create_train_loader, create_test_loader
import config

# Get runtime device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialise model
net = ResNet101(config.IMAGE_CATEGORY).to(device)
# Define optimizer
optimizer = optim.Adam(net.parameters())
# Define loss function
criterion = nn.CrossEntropyLoss().to(device)
# Get training loader and test loader
train_loader = create_train_loader()
test_loader = create_test_loader()

print("Start training......")
for epoch in range(config.EPOCHS):
    # Set the model to training mode
    net.train()
    total_loss = 0.0
    for index, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)  # Correct order of arguments

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"[{epoch + 1} / {config.EPOCHS}], average loss: {total_loss / len(train_loader)}")

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

print("Training Completed!")

# Save model
torch.save(net, "resnet101-plant-disease-recognition.pt")