import os
import zipfile
import shutil
from tqdm import tqdm
import config


def extract_dataset(zip_path, extract_to_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError("No dataset found, please use 'kaggle datasets download -d rashikrahmanpritom/plant-disease-recognition-dataset' to download the dataset.")
    if not os.path.exists(extract_to_path):
        os.mkdir(extract_to_path)
    # Extract the dataset from ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_f:
        # Get the list of files in the ZIP archive
        zip_infos = zip_f.infolist()
        
        # Initialize tqdm for visualizing the extraction progress
        with tqdm(total=len(zip_infos), unit='file') as pbar:
            for zip_info in zip_infos:
                zip_f.extract(zip_info, extract_to_path)
                pbar.update(1)
    return

def adjust_folder_structure(folder_path):
    if not os.path.exists(folder_path):
        raise IOError("Can find the base folder path!")
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Iterate over the second-layer directories
            for subsubdir in os.listdir(subdir_path):
                subsubdir_path = os.path.join(subdir_path, subsubdir)
                
                # Check if it's a directory
                if os.path.isdir(subsubdir_path):
                    # Move all files from the second-layer directory to the first-layer directory
                    for file_name in os.listdir(subsubdir_path):
                        src_file = os.path.join(subsubdir_path, file_name)
                        dst_file = os.path.join(folder_path, subsubdir, file_name)
                        
                        # Ensure the destination directory exists
                        if not os.path.exists(os.path.join(folder_path, subsubdir)):
                            os.makedirs(os.path.join(folder_path, subsubdir))
                        
                        shutil.move(src_file, dst_file)
                    
                    # Remove the now-empty second-layer directory
                    shutil.rmtree(subsubdir_path)
    # The last step, remove all empty folders 
    folders = os.listdir(folder_path)
    for folder in folders:
        subfolder_path = os.path.join(folder_path, folder)
        if not any(os.scandir(subfolder_path)):
            os.rmdir(subfolder_path)


if __name__ == "__main__":
    extract_dataset(config.ZIP_PATH, config.DATASET_PATH)
    # list all subfolders under dataset folder
    for folder in os.listdir(config.DATASET_PATH):
        path = os.path.join(config.DATASET_PATH, folder)
        # Only operate on folders.
        if os.path.isdir(path):
            adjust_folder_structure(path)