import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
### -------------------------------------------------- ###
## Script to generate occupancy grid dataset ##
# # Load mesh

DATA_DIR = "./r1.0.1/reconstruction"
SAVE_DIR = "./r1.0.1/reconstruction_occgrid"
for file in os.listdir(DATA_DIR) :
    if file.endswith(".obj") :
        
        file_path = DATA_DIR + "/" + file
        file_name = SAVE_DIR + "/" + file[:-4] + ".npy"
        mesh = trimesh.load(file_path)

        mesh.apply_translation(-mesh.centroid)
        scale = 0.95 / np.max(mesh.extents)
        mesh.apply_scale(scale)

        # mesh.show()
        # Voxelization
        target_resolution = 64
        pitch = 1.0 / target_resolution
        voxelized = mesh.voxelized(pitch=pitch)
        solid = voxelized.fill()
        vox = solid.matrix.astype(bool)  # shape: (Z, Y, X)

        # Get input voxel shape
        D, H, W = vox.shape
        print(f"Original voxel shape: {vox.shape}")

        # Prepare empty target grid
        padded = np.zeros((target_resolution, target_resolution, target_resolution), dtype=bool)

        # Compute valid region sizes (no overflow)
        dz = min(D, target_resolution)
        dy = min(H, target_resolution)
        dx = min(W, target_resolution)

        # Compute centered placement offsets
        z_offset = (target_resolution - dz) // 2
        y_offset = (target_resolution - dy) // 2
        x_offset = (target_resolution - dx) // 2

        # Final shape checks before insertion
        print(f"Target insert shape: {(dz, dy, dx)}")
        print(f"Inserting at: z[{z_offset}:{z_offset+dz}], y[{y_offset}:{y_offset+dy}], x[{x_offset}:{x_offset+dx}]")

        # Safe slice and assign
        padded[z_offset:z_offset+dz, y_offset:y_offset+dy, x_offset:x_offset+dx] = vox[:dz, :dy, :dx]

        # Add channel dimension → final shape: (64, 64, 64, 1)
        voxel_grid = np.expand_dims(padded, axis=-1)
        print("Final voxel grid shape:", voxel_grid.shape)

        np.save(file_name, voxel_grid)

        print(file_name, " Converted and Saved")

### -------------------------------------------------- ###


### -------------------------------------------------- ###
## Example --> Generate occuapncy grid for single mesh
# mesh = trimesh.load("./r1.0.1/reconstruction/20203_7e31e92a_0000_0007.obj")

# mesh.apply_translation(-mesh.centroid)
# scale = 0.95 / np.max(mesh.extents)
# mesh.apply_scale(scale)

# mesh.show()
# # Voxelization
# target_resolution = 64
# pitch = 1.0 / target_resolution
# voxelized = mesh.voxelized(pitch=pitch)
# solid = voxelized.fill()
# vox = solid.matrix.astype(bool)  # shape: (Z, Y, X)

# # Get input voxel shape
# D, H, W = vox.shape
# print(f"Original voxel shape: {vox.shape}")

# # Prepare empty target grid
# padded = np.zeros((target_resolution, target_resolution, target_resolution), dtype=bool)

# # Compute valid region sizes (no overflow)
# dz = min(D, target_resolution)
# dy = min(H, target_resolution)
# dx = min(W, target_resolution)

# # Compute centered placement offsets
# z_offset = (target_resolution - dz) // 2
# y_offset = (target_resolution - dy) // 2
# x_offset = (target_resolution - dx) // 2

# # Final shape checks before insertion
# print(f"Target insert shape: {(dz, dy, dx)}")
# print(f"Inserting at: z[{z_offset}:{z_offset+dz}], y[{y_offset}:{y_offset+dy}], x[{x_offset}:{x_offset+dx}]")

# # Safe slice and assign
# padded[z_offset:z_offset+dz, y_offset:y_offset+dy, x_offset:x_offset+dx] = vox[:dz, :dy, :dx]

# # Add channel dimension → final shape: (64, 64, 64, 1)
# voxel_grid = np.expand_dims(padded, axis=-1)
# print("Final voxel grid shape:", voxel_grid.shape)

# np.save("/home/mmpug/Desktop/CADSTUFF/L3DPROJ/r1.0.1/reconstruction_occgrid/occgrid.npy", voxel_grid)
### -------------------------------------------------- ###

### -------------------------------------------------- ###
# Example visualization

# occgrid = np.load("./reconstruction_occgrid/148133_3050ab1d_0000.npy")
# voxels = occgrid.squeeze()

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot solid voxels
# ax.voxels(voxels, facecolors='royalblue', edgecolor='k', linewidth=0.2)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Solid Voxel Grid')
# plt.tight_layout()
# plt.show()

### -------------------------------------------------- ###
