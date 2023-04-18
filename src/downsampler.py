import zipfile
import patoolib
import os
import shutil
import random
from torchvision.datasets import HMDB51
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.io as io
import rarfile
import random

class HMBD51Downsampler:
  
  def __init__(self, rar_file_origin_path, rar_file_destination_path, destination_path, percentage):
    
    self._extract_HMDB51_rar(rar_file_origin_path, rar_file_destination_path) 

    unrared_folder_path = rar_file_destination_path + "/HMBD51"
    self._extract_rar(unrared_folder_path)
    self._downsample_HMDB51(unrared_folder_path, destination_path, percentage)

  def _extract_HMDB51_rar(self, origin_path, destination_path):
    
    folder_path = destination_path # destination sub-folder to hold the data.
    subfolder_name = "HMBD51"
    subfolder_path = os.path.join(folder_path, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    patoolib.extract_archive(origin_path, outdir=subfolder_path)
  
  def _extract_rar(self, folder_path):

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a rar file
        if filename.endswith(".rar"):
            # Open the rar file and extract its contents
            with rarfile.RarFile(os.path.join(folder_path, filename)) as rf:
                rf.extractall(path=os.path.join(folder_path, os.path.splitext(filename)[0]))
            # Delete the rar file
            os.remove(os.path.join(folder_path, filename))

  def _downsample_HMDB51(self, origin_directory, destination_directory, percentage):

    # Get all classes
    classes = [d for d in os.listdir(origin_directory) if os.path.isdir(os.path.join(origin_directory, d))]

    # Loop through each class
    for class_name in classes:
        # Create a directory for the current class
        os.makedirs(os.path.join(destination_directory, class_name), exist_ok=True)
        
        # Get all videos of the current class
        class_dir = os.path.join(origin_directory, class_name, class_name)
        videos = os.listdir(class_dir)
        
        # Calculate the number of videos to extract
        num_videos = round(len(videos) * percentage)
        
        # Shuffle the list of videos
        random.shuffle(videos)
        
        # Select the required number of videos
        selected_videos = videos[:num_videos]
        
        # Copy the selected videos to the smaller dataset directory
        for video_name in selected_videos:
            video_path = os.path.join(class_dir, video_name)
            shutil.copy(video_path, os.path.join(destination_directory, class_name))