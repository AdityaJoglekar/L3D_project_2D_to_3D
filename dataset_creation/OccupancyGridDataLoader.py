import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import io
import cairosvg
import torchvision.transforms as transforms

class OccupancyGridDataset(Dataset):
    def __init__(self, root_dir, transform=None, grid_transform=None):
        self.occupancy_dir = os.path.join(root_dir, 'occupancy')
        self.views_dir = os.path.join(root_dir, 'svgs')
        self.transform = transform
        self.grid_transform = grid_transform

        # Base names from .npy files
        self.sample_ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.occupancy_dir)
            if f.endswith('.npy')
        ]
        self.sample_ids.sort()

        self.view_suffixes = ['front', 'right', 'top']

    def __len__(self):
        return len(self.sample_ids)

    def _load_svg_as_image(self, path):
        """Converts an SVG file to a PIL Image."""
        png_bytes = cairosvg.svg2png(url=path)
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return image

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Load occupancy grid
        grid_path = os.path.join(self.occupancy_dir, f"{sample_id}.npy")
        grid = np.load(grid_path)
        if self.grid_transform:
            grid = self.grid_transform(grid)
        else:
            grid = torch.tensor(grid, dtype=torch.float32)

        # Load 3 views
        views = []
        for view_type in self.view_suffixes:
            view_path = os.path.join(self.views_dir, f"{sample_id}_{view_type}.svg")
            view = self._load_svg_as_image(view_path)
            if self.transform:
                view = self.transform(view)
            else:
                view = transforms.ToTensor()(view)
            views.append(view)

        views = torch.stack(views, dim=0)  # shape: [3, C, H, W]

        return {
            'occupancy': grid,
            'views': views,
            'id': sample_id
        }

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

view_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = OccupancyGridDataset("/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/filtered", transform=view_transform)
dataloader = DataLoader(dataset, batch_size=2)

for batch in dataloader:
    print("BATCH ID : ", batch['id'])                         # list of sample names
    print("OCCUPANCY : ", batch['occupancy'].shape)            # e.g. [B, D, H, W]
    print("VIEWS SHAPE : ", batch['views'].shape)                # e.g. [B, 3, 3, 128, 128]


## Filtered dataset creation
# import os
# import shutil

# # Original paths
# svg_dir = '/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/svg_outputs'
# occ_dir = '/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/occupancy_net/reconstruction_occgrid'

# # Destination paths
# filtered_occ_dir = '/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/filtered/occupancy'
# filtered_svg_dir = '/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/filtered/svgs'

# # Create destination folders if they don't exist
# os.makedirs(filtered_occ_dir, exist_ok=True)
# os.makedirs(filtered_svg_dir, exist_ok=True)

# # List files
# svg_files = set(os.listdir(svg_dir))
# occupancy_files = [f for f in os.listdir(occ_dir) if f.endswith('.npy')]

# required_views = ['front', 'right', 'top']

# count = 0

# for occ_file in occupancy_files:
#     base_name = os.path.splitext(occ_file)[0]

#     # Check if all 3 svg views exist
#     all_views_exist = all(f"{base_name}_{view}.svg" in svg_files for view in required_views)

#     if all_views_exist:
#         # Copy occupancy file
#         src_occ_path = os.path.join(occ_dir, occ_file)
#         dst_occ_path = os.path.join(filtered_occ_dir, occ_file)
#         shutil.copy(src_occ_path, dst_occ_path)

#         # Copy svg views
#         for view in required_views:
#             svg_name = f"{base_name}_{view}.svg"
#             src_svg_path = os.path.join(svg_dir, svg_name)
#             dst_svg_path = os.path.join(filtered_svg_dir, svg_name)
#             shutil.copy(src_svg_path, dst_svg_path)

#         count += 1

# print(f" Copied {count} valid samples with all 3 views.")
