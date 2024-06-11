import torch
from torchvision.transforms import v2  
from PIL import Image
import config
import os
import argparse


def preprocess_image(image_path):
    transform = v2.Compose([
    v2.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
])
    image = Image.open(image_path)
    image = transform(image)
    return image


def predict(image_path, model, device, classes):
    input_tensor = preprocess_image(image_path).to(device)
    # Add one dimension
    input_tensor = input_tensor.unsqueeze(0)
    # Predict category
    output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    return classes[predicted_class.item()]


def main(folder_path):
    # Get runtime device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pretrained model
    net = torch.load("./models/resnet50-plant-disease-recognition.pt").to(device)
    # Define classes
    classes = ("Healthy", "Powdery", "Rust")
    
    # Iterate over all image files in the folder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        if os.path.isfile(image_path):  # Ensure it is a file
            prediction = predict(image_path, net, device, classes)
            print(f"Image: {image_file}, Prediction: {prediction}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Recognition")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing image files")
    args = parser.parse_args()

    main(args.folder_path)