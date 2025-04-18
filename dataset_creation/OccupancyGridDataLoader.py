import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import io
import cairosvg
import torchvision.transforms as transforms

class OccupancyGridDataset(Dataset):
    def __init__(self, occupancy_dir, svg_dir, transform=None, grid_transform=None):
        self.occupancy_dir = occupancy_dir
        self.svg_dir = svg_dir
        self.transform = transform
        self.grid_transform = grid_transform

        self.samples = []

        # Create a mapping from occupancy base name to full path
        occ_map = {
            os.path.splitext(f)[0]: os.path.join(occupancy_dir, f)
            for f in os.listdir(occupancy_dir)
            if f.endswith('.npy')
        }

        # Go through SVGs and associate with occupancy
        for svg_file in os.listdir(svg_dir):
            if not svg_file.endswith('.svg'):
                continue

            # Parse base name and view direction
            parts = os.path.splitext(svg_file)[0].rsplit('_', 1)
            if len(parts) != 2:
                continue
            base_name, view_type = parts

            if base_name in occ_map:
                self.samples.append({
                    'occ_path': occ_map[base_name],
                    'svg_path': os.path.join(svg_dir, svg_file),
                    'base_name': base_name,
                    'view_type': view_type
                })

        self.samples.sort(key=lambda x: (x['base_name'], x['view_type']))

    def __len__(self):
        return len(self.samples)

    def _load_svg_as_image(self, path):
        """Convert SVG file to PIL Image."""
        png_bytes = cairosvg.svg2png(url=path)
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load occupancy grid
        grid = np.load(sample['occ_path'])
        if self.grid_transform:
            grid = self.grid_transform(grid)
        else:
            grid = torch.tensor(grid, dtype=torch.float32)

        # Load single view
        view = self._load_svg_as_image(sample['svg_path'])
        if self.transform:
            view = self.transform(view)
        else:
            view = transforms.ToTensor()(view)

        return {
            'occupancy': grid,
            'view': view,
            'view_type': sample['view_type'],
            'id': sample['base_name']
        }

## ----------- Example Use ----------- ##

# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms

# view_transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

# dataset = OccupancyGridDataset(
#     occupancy_dir='/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/filtered/occupancy',
#     svg_dir='/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/dataset/filtered/svgs',
#     transform=transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])
# )


# dataloader = DataLoader(dataset, batch_size=2)
# for batch in dataloader:
#     print("BATCH ID : ", batch['id'])                         # list of sample names
#     print("OCCUPANCY : ", batch['occupancy'].shape)            # e.g. [B, D, H, W]
#     print("VIEWS SHAPE : ", batch['view'].shape)                # e.g. [B, 3, 128, 128]
#     print("VIEW TYPE : ", batch['view_type'])

## ----------- END Example Use ----------- ##

## ----------- Filtered dataset creation ----------- ##
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

## ----------- END Filtered dataset creation ----------- ##