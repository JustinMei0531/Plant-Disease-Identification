# Dataset paths
ZIP_PATH = "plant-disease-recognition-dataset.zip"
DATASET_PATH = "./dataset"

# Input image size
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
# The number of image categories
IMAGE_CATEGORY = 3
# Batch size
TRAINING_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 2

# Training parameters
EPOCHS = 10
LEARNING_RATE = 1e-5

# Models path
RESNET34_PARTS_PATH = "./models/resnet34"
RESNET50_PARTS_PATH = "./models/resnet50"
RESNET101_PARTS_PATH = "./models/resnet101"

RESNET34_MODEL_PATH = "./models/resnet34/resnet34-plant-disease-recognition.pt"
RESNET50_MODEL_PATH = "./models/resnet50/resnet50-plant-disease-recognition.pt"
RESNET101_MODEL_PATH = "./models/resnet101/resnet101-plant-disease-recognition.pt"