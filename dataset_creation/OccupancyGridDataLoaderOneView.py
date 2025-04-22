import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class OccupancyGridDataset(Dataset):
    def __init__(self, occupancy_dir, image_dir, split_path, split_name='train', transform=None, grid_transform=None):
         self.occupancy_dir = occupancy_dir
         self.image_dir = image_dir
         self.transform = transform
         self.grid_transform = grid_transform
 
         self.samples = []
 
        # Load split.json and extract relevant filenames
         with open(split_path, 'r') as f:
             split_data = json.load(f)
         allowed_filenames = set(split_data[split_name])  # Just base names, no extension
 
         # Map occupancy filenames (base name -> path)
         occ_map = {
             os.path.splitext(f)[0]: os.path.join(occupancy_dir, f)
             for f in os.listdir(occupancy_dir)
            if f.endswith('.npy') and os.path.splitext(f)[0] in allowed_filenames
         }


         # Go through PNGs and associate with occupancy
         for img_file in os.listdir(image_dir):
             if not img_file.endswith('.png'):
                 continue
 
             # Parse base name and view direction
             parts = os.path.splitext(img_file)[0].rsplit('_', 1)
             if len(parts) != 2:
                 continue
             base_name, view_type = parts
 
             if base_name in occ_map:
                 self.samples.append({
                     'occ_path': occ_map[base_name],
                     'img_path': os.path.join(image_dir, img_file),
                     'base_name': base_name,
                     'view_type': view_type
                 })
 
         self.samples.sort(key=lambda x: (x['base_name'], x['view_type']))
 
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load occupancy grid
        grid = np.load(sample['occ_path'])
        if self.grid_transform:
            grid = self.grid_transform(grid)
        else:
            grid = torch.tensor(grid, dtype=torch.float32)

        # Load image view
        view = Image.open(sample['img_path']).convert("RGB")
        if self.transform:
            view = self.transform(view)
        else:
            view = transforms.ToTensor()(view)

        view = view.permute(1, 2, 0)  # HWC format if needed

        return {
            'occupancy': grid,
            'view': view,
            'view_type': sample['view_type'],
            'id': sample['base_name']
        }
 
