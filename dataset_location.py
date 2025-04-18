# specify the root location where u downloaded the dataset
root_location = "../"

use_full_dataset = False
use_f360_dataset = True
dataset_name = (
    "r2n2_shapenet_dataset_full" if use_full_dataset else "r2n2_shapenet_dataset"
)

if use_f360_dataset :
    root_location = "./dataset"
    dataset_name = "fusion360_dataset"

R2N2_PATH = f"{root_location}/{dataset_name}/r2n2"
SHAPENET_PATH = f"{root_location}/{dataset_name}/shapenet"
FUSION_360_OCCUPANCY_PATH = f"{root_location}/{dataset_name}/occupancy"
FUSION_360_VIEWS_PATH = f"{root_location}/{dataset_name}/pngs"

if use_full_dataset:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_3c.json"  # split file contains data entry for 3 classes
else:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_03001627.json"  # split file contains data entry for 03001627 class

if use_f360_dataset:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_f360.json"

## ---------- TEST IF CORRECT VARIABLES ARE BEING LOADED ----------------- ## 
# print(use_f360_dataset)
# print(root_location)
# print(dataset_name)
# print(FUSION_360_OCCUPANCY_PATH)
# print(FUSION_360_VIEWS_PATH)
# print(SPLITS_PATH)
# import os
# print(os.listdir(FUSION_360_OCCUPANCY_PATH))
## ---------- END TEST IF CORRECT VARIABLES ARE BEING LOADED ----------------- ##