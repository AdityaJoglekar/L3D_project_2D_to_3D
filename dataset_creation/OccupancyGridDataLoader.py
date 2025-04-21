import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OccupancyGridDataset(Dataset):
    def __init__(self, occupancy_dir, image_dir, split_path, split_name='train', transform=None, grid_transform=None):
        self.occupancy_dir = occupancy_dir
        self.image_dir = image_dir
        self.transform = transform
        self.grid_transform = grid_transform
        self.samples = []

        try:
            with open(split_path, 'r') as f:
                split_data = json.load(f)
            allowed_base_names = set(split_data[split_name])
        except FileNotFoundError:
            print(f"Error: Split file not found at {split_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {split_path}")
            return

        occ_map = {
            os.path.splitext(f)[0]: os.path.join(occupancy_dir, f)
            for f in os.listdir(occupancy_dir)
            if f.endswith('.npy') and os.path.splitext(f)[0] in allowed_base_names
        }

        image_groups = {}
        for img_file in os.listdir(image_dir):
            if not img_file.endswith('.png'):
                continue
            parts = os.path.splitext(img_file)[0].rsplit('_', 1)
            if len(parts) == 2:
                base_name, view_type = parts
                if base_name in allowed_base_names:
                    if base_name not in image_groups:
                        image_groups[base_name] = {}
                    image_groups[base_name][view_type] = os.path.join(image_dir, img_file)

        for base_name, occ_path in occ_map.items():
            if base_name in image_groups and 'front' in image_groups[base_name] and 'right' in image_groups[base_name] and 'top' in image_groups[base_name]:
                self.samples.append({
                    'occ_path': occ_path,
                    'img_paths': image_groups[base_name],
                    'base_name': base_name
                })

        self.samples.sort(key=lambda x: x['base_name'])

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

        views = {}
        for view_type, img_path in sample['img_paths'].items():
            # Load image view
            view = Image.open(img_path).convert("RGB")
            if self.transform:
                view = self.transform(view)
            else:
                view = transforms.ToTensor()(view)
            views[view_type] = view.permute(1, 2, 0)  # HWC format

        return {
            'occupancy': grid,
            'front_view': views.get('front'),
            'side_view': views.get('right'),
            'top_view': views.get('top')
        }

def custom_collate_fn(batch):
    occupancy = torch.stack([item['occupancy'] for item in batch])
    front_view = torch.stack([item['front_view'] for item in batch])
    side_view = torch.stack([item['side_view'] for item in batch])
    top_view = torch.stack([item['top_view'] for item in batch])
    return {
        'voxels': occupancy,  # renaming to match your training script
        'front_images': front_view,
        'side_images': side_view,  
        'top_images': top_view,    
    }
## ----------- Example Use ----------- ##



# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms

# view_transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

# print("here")
# dataset = OccupancyGridDataset(
#     occupancy_dir='/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/fusion360_dataset/occupancy',
#     image_dir='/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/fusion360_dataset/pngs',
#     split_path='/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/fusion360_dataset/splits_f360.json',
#     split_name='train',
#     transform=transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])
# )

# dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)
# for batch in dataloader:
#     print("here2")                   # list of sample names
#     print("OCCUPANCY SHAPE : ", batch['occupancy'].shape)            # e.g. [B, D, H, W]
#     print("FRONT VIEW SHAPE : ", batch['front_images'].shape if batch['front_images'] is not None else None) # e.g. [B, H, W, 3]
#     print("SIDE VIEW SHAPE : ", batch['side_images'].shape if batch['side_images'] is not None else None)   # e.g. [B, H, W, 3]
#     print("TOP VIEW SHAPE : ", batch['top_images'].shape if batch['top_images'] is not None else None)     # e.g. [B, H, W, 3]
#     break 