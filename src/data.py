
from torch.utils.data import Dataset, DataLoader 
import pathlib
import os
from PIL import Image
# import numpy as np

def create_dataloaders(training_directory: str, test_directory: str, transform, batch_size: int, num_workers: int = os.cpu_count()):
  #dataset = HMDB51Dataset("/content/drive/MyDrive/thesis/data/HMDB-51-downsampled_copy", transform=backbone_transform) 
  training_data = HMDB51Dataset(training_directory, transform=transform) 
  test_data = HMDB51Dataset(test_directory, transform=transform) 
  
  class_names = training_data.classes
  
  training_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
  test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
  
  return training_dataloader, test_dataloader, class_names

class HMDB51Dataset(Dataset):
  def __init__(self, targ_dir: str, transform=None):
    self.total_examples = self.count_subfolders(targ_dir)
    self.classes, self.class_to_index = self.get_classes(targ_dir)
    self.paths = self.get_paths(targ_dir)
    self.transform = transform

  def __len__(self):
    return self.total_examples

  def __getitem__(self, index: int):
    X = self.get_example(self.paths[index])
    y = self.get_label(self.paths[index])
    return X, y

  def count_subfolders(self, folder_path):
      count = 0
      for folder in os.listdir(folder_path):
          subfolder_path = os.path.join(folder_path, folder)
          if os.path.isdir(subfolder_path):
              for subfolder in os.listdir(subfolder_path):
                  if os.path.isdir(os.path.join(subfolder_path, subfolder)):
                      count += 1
      return count

  def get_classes(self, targ_dir: str):

      count = 0
      classes = []
      indexes = []

      for folder in os.listdir(targ_dir):
          subfolder_path = os.path.join(targ_dir, folder)
          classes.append(folder)
          indexes.append(count)
          count += 1
      return classes, indexes

  def get_paths(self, targ_dir: str):
    path_obj = pathlib.Path(targ_dir)
    paths = []

    for subfolder in path_obj.iterdir():
        if subfolder.is_dir():      
            for subsubfolder in subfolder.iterdir():
                if subsubfolder.is_dir():
                    paths.append(subsubfolder.resolve().as_posix())
    return paths
  
  def get_label(self, directory: str):
    return self.classes.index(pathlib.Path(directory).parts[-2])

  def get_example(self, directory: str):
    segments = []
    segment_prefix = "/segment-"
    FLOW_FRAME_NAME = "flow_frame.png"
    RGB_FRAME_NAME = "rgb_frame.png"
    POSE_FRAME_NAME = "pose_frame.png"
    SEGMENTS_NUMBER = self.get_subfolders_number(directory)
    
    for segment_index in range(SEGMENTS_NUMBER):
      segment_path = directory + segment_prefix + str(segment_index) + "/"
      rgb_frame_path = segment_path + RGB_FRAME_NAME
      flow_frame_path = segment_path + FLOW_FRAME_NAME
      pose_frame_path = segment_path + POSE_FRAME_NAME
      
      # Solution for RGB convertion from https://stackoverflow.com/questions/59218671/runtimeerror-output-with-shape-1-224-224-doesnt-match-the-broadcast-shape
      rgb_image = Image.open(rgb_frame_path).convert('RGB')
      #print(f" RGB Dimensions:{np.asarray(rgb_image).shape}\n")
      rgb_frame = self.transform(rgb_image)
      
      flow_image = Image.open(flow_frame_path).convert('RGB') 
      #print(f"Flow Dimensions:{np.stack((np.asarray(flow_image),)*3, axis=-1).shape}\n")
      flow_frame = self.transform(flow_image)

      pose_image = Image.open(pose_frame_path).convert('RGB')
      #print(f"Pose Dimensions:{np.asarray(pose_image).shape}\n")
      pose_frame = self.transform(pose_image)
      
      segments.append((rgb_frame, flow_frame, pose_frame))

    return segments

  def get_subfolders_number(self, directory):
    path_obj = pathlib.Path(directory)

    num_subfolders = 0
    for subfolder in path_obj.iterdir():
        if subfolder.is_dir():
            num_subfolders += 1
    return num_subfolders

# dataset = HMDB51Dataset("/content/drive/MyDrive/thesis/data/HMDB-51-downsampled_copy", transform=backbone_transform) 
# dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=os.cpu_count(), shuffle=True)
# video_batch, label_batch = next(iter(dataloader))
