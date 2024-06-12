import os
import config


def split_file(file_path, chunk_size):
    with open(file_path, 'rb') as f:
        chunk_number = 0
        while chunk := f.read(chunk_size):
            with open(f"{file_path}.part{chunk_number}", 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_number += 1

def combine_files(output_file_path, input_file_paths):
    with open(output_file_path, 'wb') as output_file:
        for file_path in input_file_paths:
            with open(file_path, 'rb') as input_file:
                output_file.write(input_file.read())

'''
Please note that please do not run this code, because the model uploaded to github has been split into several parts.
'''
# Split the model, each path is 10Mb
# models_path = (
#     "./models/resnet34/resnet34-plant-disease-recognition.pt",
#     "./models/resnet50/resnet50-plant-disease-recognition.pt",
#     "./models/resnet101/resnet101-plant-disease-recognition.pt"
# )

# for path in models_path:
#     split_file(path, 10 * 1024 * 1024)

combine_files(os.path.join(config.RESNET34_PARTS_PATH, "resnet34-plant-disease-recognition.pt"),  
              [os.path.join(config.RESNET34_PARTS_PATH, path) for path in os.listdir(config.RESNET34_PARTS_PATH)])
combine_files(os.path.join(config.RESNET50_PARTS_PATH, "resnet50-plant-disease-recognition.pt"),  
              [os.path.join(config.RESNET50_PARTS_PATH, path) for path in os.listdir(config.RESNET50_PARTS_PATH)])
combine_files(os.path.join(config.RESNET101_PARTS_PATH, "resnet101-plant-disease-recognition.pt"),  
              [os.path.join(config.RESNET101_PARTS_PATH, path) for path in os.listdir(config.RESNET101_PARTS_PATH)])