import os
import shutil
from pathlib import Path

def copy_images_recursive(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    # print('here')
    # Walk through the source directory and its subdirectories
    # print(source_dir)
    for root, dirs, files in os.walk(source_dir):
        # print(root)
        for file in files:
            # Filter only image files (you can extend this list based on your image formats)
            if file.lower().endswith(('instanceids.png')):
                
                source_path = os.path.join(root, file)
                # Adjust the destination path to maintain the directory structure
                # destination_path = os.path.join(destination_dir, os.path.relpath(source_path, source_dir))

                # Use shutil.copy to perform the actual copy
                shutil.copy(source_path, destination_dir)
                print(f"Copied: {source_path} to {destination_dir}")

if __name__ == "__main__":
    # Specify the source and destination directories
    source_directory = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest/gtFine/train/"
    destination_directory = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest/DB_New/trainannotnn/"

    # Call the function to copy images
    copy_images_recursive(source_directory, destination_directory)
