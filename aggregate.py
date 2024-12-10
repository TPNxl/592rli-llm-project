import os
import shutil
import torch

def clean_folder(folder: str):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# This function deletes all the files in output_folder,
# takes all the files in subfolders ending in ".hlst" at the path datasets_folder 
# and copies them at the base level to the output_folder
# Make sure to copy them and not move them
def merge_datasets_folder(folder: str, output_folder: str):
    clean_folder(output_folder)
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".hlst"):
                full_path = os.path.join(root, file)
                epoch = root.split("/")[-1].split("epoch_")[-1]
                new_name = f"{epoch}_{file}"
                new_path = os.path.join(output_folder, new_name)
                shutil.copy(full_path, new_path)


if __name__ == "__main__":
    clean_folder("complete_dataset")
    merge_datasets_folder("datasets", "complete_dataset")