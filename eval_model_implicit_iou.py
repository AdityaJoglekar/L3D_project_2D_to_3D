import argparse
import time
import torch
# from model import SingleViewto3D
from model_implicit import SingleViewto3D
# from model_implicit_PerceiverAdaLN import SingleViewto3D
# from model_implicit_OccNet import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import numpy as np
from dataset_creation.OccupancyGridDataLoader import OccupancyGridDataset
import pytorch3d
from utils import get_device, get_mesh_renderer, get_points_renderer
from PIL import Image
import imageio
import numpy as np
from matplotlib import cm, colors
from matplotlib.colors import LightSource
import torchvision.transforms as transforms

def custom_collate_fn(batch):
    occupancy = torch.stack([item['occupancy'] for item in batch])
    view = torch.stack([item['view'] for item in batch])
    return {
        'voxels': occupancy,  # renaming to match your training script
        'images': view,
    }

def plot_fields(xyz, field, titles = None, cmap="coolwarm", filename = "vox.png"):

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # scatter = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=field, cmap=cmap)
    scatter = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=field, s=field*100, alpha = 0.3, cmap=cmap)
    ax.view_init(30, 30,vertical_axis='y')
    cbar = fig.colorbar(scatter)
    plt.savefig(filename)

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=200, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda:0', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument("--num_samples", default=32*32*32, type=int)
    parser.add_argument("--model_name", default="PerceiverAdaLN", type=str)
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    indices = torch.randperm(64*64*64)[:args.num_samples]  
    images = feed_dict['images'].squeeze(1)
    voxels_gt = feed_dict['voxels'].reshape(args.batch_size,-1)[:,indices] 
    voxels_gt = voxels_gt.reshape(args.batch_size,1,32,32,32)
    H,W,D = voxels_gt.shape[2:]
    vertices_src, faces_src = mcubes.marching_cubes(voxels_gt.detach().cpu().squeeze().numpy(), isovalue=0.5)
    vertices_src = torch.tensor(vertices_src).float()
    faces_src = torch.tensor(faces_src.astype(int))
    mesh_gt = pytorch3d.structures.Meshes([vertices_src], [faces_src]) 
    
    return images.to(args.device), mesh_gt, indices, voxels_gt

# def preprocess(feed_dict, args):
#     for k in ['images']:
#         feed_dict[k] = feed_dict[k].to(args.device)

#     images = feed_dict['images'].squeeze(1)
#     print(feed_dict['voxels'].shape )
#     # Get voxel grid shape
#     B, D, H, W , C= feed_dict['voxels'].shape  # assuming (B, 1, 64, 64, 64)
#     assert D == H == W == 64

#     # Generate voxel center coordinates in normalized space [0, 1] or [-1, 1]
#     grid = torch.stack(torch.meshgrid(
#         torch.linspace(0.5 / D, 1 - 0.5 / D, D),
#         torch.linspace(0.5 / H, 1 - 0.5 / H, H),
#         torch.linspace(0.5 / W, 1 - 0.5 / W, W),
#         indexing='ij'  # To get (Z, Y, X) order
#     ), dim=-1)  # Shape: (D, H, W, 3)

#     grid = grid.reshape(-1, 3)  # Shape: (64*64*64, 3)

#     # Sample a subset of centers
#     indices = torch.randperm(D * H * W)[:args.num_samples]
#     sampled_points = grid[indices]  # Shape: (num_samples, 3)

#     # Reshape voxel grid for GT occupancy values
#     voxels = feed_dict['voxels'].reshape(args.batch_size, -1)
#     voxels_gt = voxels[:, indices]
#     voxels_gt = voxels_gt.reshape(args.batch_size, 1, 32, 32, 32)

#     return images.to(args.device), sampled_points, indices, voxels_gt


def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    # plt.savefig(f'eval_{args.type}', bbox_inches='tight')
    plt.savefig(f'eval_{args.type}_{args.model_name}_implicit', bbox_inches='tight')


# def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
#     metrics = {}
#     lengths_pred = torch.full(
#         (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
#     )
#     lengths_gt = torch.full(
#         (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
#     )

#     # For each predicted point, find its neareast-neighbor GT point
#     knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
#     # Compute L1 and L2 distances between each pred point and its nearest GT
#     pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
#     pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

#     # For each GT point, find its nearest-neighbor predicted point
#     knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
#     # Compute L1 and L2 dists between each GT point and its nearest pred point
#     gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
#     gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

#     # Compute precision, recall, and F1 based on L2 distances
#     for t in thresholds:
#         precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
#         recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
#         f1 = (2.0 * precision * recall) / (precision + recall + eps)
#         metrics["Precision@%f" % t] = precision
#         metrics["Recall@%f" % t] = recall
#         metrics["F1@%f" % t] = f1

#     # Move all metrics to CPU
#     metrics = {k: v.cpu() for k, v in metrics.items()}
#     return metrics

# def evaluate(predictions, mesh_gt, thresholds, args):
#     if args.type == "vox":
#         voxels_src = predictions.reshape(args.batch_size,1,32,32,32)
#         # print('voxels_src',voxels_src.detach().cpu().numpy()[np.where(voxels_src.detach().cpu().numpy()>0.5)])
#         # print('voxels_src',len(voxels_src.detach().cpu().numpy()[np.where(voxels_src.detach().cpu().numpy()>0.5)]))
#         H,W,D = voxels_src.shape[2:]
#         vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
#         # if vertices_src.shape == torch.Size([0, 3]):
#             # print('here')
#             # print('vertices',vertices_src[np.where(vertices_src>-1000)])
#             # vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.2)
#         vertices_src = torch.tensor(vertices_src).float()
#         faces_src = torch.tensor(faces_src.astype(int))
#         mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src]) 
#         pred_points = sample_points_from_meshes(mesh_src, args.n_points)
#         pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
#         # Apply a rotation transform to align predicted voxels to gt mesh
#         angle = -math.pi
#         axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
#         Rot = axis_angle_to_matrix(axis_angle)
#         T_transform = Rotate(Rot)
#         pred_points = T_transform.transform_points(pred_points)
#         # re-center the predicted points
#         pred_points = pred_points - pred_points.mean(1, keepdim=True)
#     # elif args.type == "point":
#     #     pred_points = predictions.cpu()
#     # elif args.type == "mesh":
#     #     pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

#     gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
#     if args.type == "vox":
#         gt_points = gt_points - gt_points.mean(1, keepdim=True)
#     metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
#     return metrics

# def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
#     metrics = {}
#     lengths_pred = torch.full(
#         (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
#     )
#     lengths_gt = torch.full(
#         (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
#     )

#     # Distances from pred → gt and gt → pred
#     knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
#     pred_to_gt_dists = knn_pred.dists[..., 0].sqrt()

#     knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
#     gt_to_pred_dists = knn_gt.dists[..., 0].sqrt()

#     for t in thresholds:
#         in_pred = (pred_to_gt_dists < t).float()  # (B, N_pred)
#         in_gt = (gt_to_pred_dists < t).float()    # (B, N_gt)

#         # Intersection: points matched in both
#         intersection = in_pred.sum(dim=1) + in_gt.sum(dim=1)

#         # Union: total unique points
#         union = pred_points.shape[1] + gt_points.shape[1]

#         iou = 100.0 * (intersection / (union + eps))  # (B,)
#         metrics["IoU@%f" % t] = iou

#     # Move all to CPU
#     metrics = {k: v.cpu() for k, v in metrics.items()}
#     return metrics

# def evaluate(predictions, mesh_gt, thresholds, args):
#     if args.type == "vox":
#         # Reshape predictions to voxel grid
#         voxels_src = predictions.reshape(args.batch_size, 1, 32, 32, 32)
#         H, W, D = voxels_src.shape[2:]

#         # Extract surface using marching cubes
#         vertices_src, faces_src = mcubes.marching_cubes(
#             voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5
#         )
#         vertices_src = torch.tensor(vertices_src).float()
#         faces_src = torch.tensor(faces_src.astype(int))
#         mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])

#         # Sample points from predicted mesh
#         pred_points = sample_points_from_meshes(mesh_src, args.n_points)
#         pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)

#         # Align to GT
#         angle = -math.pi
#         axis_angle = torch.tensor([[0.0, angle, 0.0]])
#         Rot = axis_angle_to_matrix(axis_angle)
#         T_transform = Rotate(Rot)
#         pred_points = T_transform.transform_points(pred_points)

#         # Center predicted points
#         pred_points = pred_points - pred_points.mean(1, keepdim=True)

#     # Ground truth sampling and centering
#     gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
#     if args.type == "vox":
#         gt_points = gt_points - gt_points.mean(1, keepdim=True)

#     # Compute IoU metrics
#     metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
#     return metrics

def evaluate(predictions, voxels_gt, thresholds, args):
    """
    Compute IoU directly from predicted and ground truth voxel grids.
    Assumes both are shaped (B, 1, D, H, W) or (B, D, H, W).
    """
    predictions = predictions.reshape( 32, 32, 32).cpu()
    voxels_gt = voxels_gt.reshape(32,32,32).cpu()
    # if predictions.dim() == 5:
    #     predictions = predictions.squeeze(1)  # (B, D, H, W)
    # if voxels_gt.dim() == 5:
    #     voxels_gt = voxels_gt.squeeze(1)

    # B = predictions.shape[0]
    metrics = {}

    for t in thresholds:
        # Threshold predicted voxels to binary
        pred_bin = (predictions > t).float()
        gt_bin = (voxels_gt > 0.5).float()

        intersection = (pred_bin * gt_bin).sum()  # Per batch
        union = ((pred_bin + gt_bin) > 0).float().sum() + 1e-8

        iou = (intersection / union)
        metrics["IoU@%f" % t] = iou.cpu()

    return metrics

# def evaluate(predictions, ground_truth) :
#     predictions = predictions.detach().cpu().numpy().reshape(32,32,32)
#     ground_truth = ground_truth.detach().cpu().numpy().reshape(32,32,32)
#     print("shapes in evaluate : ", predictions.shape, ground_truth.shape)
#     intersection = np.logical_and(predictions, ground_truth).sum()
#     union = np.logical_or(predictions, ground_truth).sum()

#     if union == 0 :
#         return 1 if intersection == 0 else 0.0
#     print(intersection/union)
    
#     return intersection / union

def evaluate_model(args):
    # r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    # loader = torch.utils.data.DataLoader(
    #     r2n2_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     collate_fn=collate_batched_R2N2,
    #     pin_memory=True,
    #     drop_last=True)
    # eval_loader = iter(loader)

    f360_dataset = OccupancyGridDataset(
        occupancy_dir=dataset_location.FUSION_360_OCCUPANCY_PATH,
        image_dir=dataset_location.FUSION_360_VIEWS_PATH,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]),
        split_path = dataset_location.SPLITS_PATH,
        split_name='test'
    )

    loader = torch.utils.data.DataLoader(
        f360_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        # checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        checkpoint = torch.load(f'/home/mmpug/Desktop/CADSTUFF/L3DPROJ/L3D_project_2D_to_3D/checkpoint_implicit_PerceiverAdaLN_99000.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        # images_gt, points_gt, indices, voxels_gt = preprocess(feed_dict, args)
        images_gt, mesh_gt, indices, voxels_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args, indices)
        xl = np.linspace(-1,1,64)
        yl = np.linspace(-1,1,64)  
        zl = np.linspace(-1,1,64)
        xc, yc, zc = np.meshgrid(xl, yl, zl, indexing='ij')
        coords = np.concatenate((xc.reshape(-1,1),yc.reshape(-1,1),zc.reshape(-1,1)),axis=1)
        print('coords[indices,:] ',coords[indices,:].shape)
        coords = torch.tensor(coords, dtype = torch.float32, device = args.device).cpu().detach()
        plot_fields(coords[indices,:],predictions.reshape(args.batch_size,-1,1)[0,:,:].cpu().detach().numpy(), filename="predvox.png")

        xl = np.linspace(-1,1,64)
        yl = np.linspace(-1,1,64)  
        zl = np.linspace(-1,1,64)
        xc, yc, zc = np.meshgrid(xl, yl, zl, indexing='ij')
        coords = np.concatenate((xc.reshape(-1,1),yc.reshape(-1,1),zc.reshape(-1,1)),axis=1)
        print('coords[indices,:] ',coords[indices,:].shape)
        coords = torch.tensor(coords, dtype = torch.float32, device = args.device).cpu().detach()
        plot_fields(coords[indices,:],voxels_gt.reshape(args.batch_size,-1,1)[0,:,:].cpu().detach().numpy(), filename="gtvox.png")

        print(predictions.shape)
        print('pred',torch.max(predictions))

        metrics = evaluate(predictions, voxels_gt, thresholds, args)
        print(metrics.keys())
        # metrics = evaluate(predictions, voxels_gt)

        # TODO:
        # if (step % args.vis_freq) == 0:
        #     # visualization block
        #     #  rend = 
        #     plt.imsave(f'vis/{step}_{args.type}.png', rend)
        if (step % args.vis_freq) == 0:
            R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=20, elev=30, azim=120)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=args.device)
            pred = predictions
            ###########  Problem 3.1  ############
            # voxels_src = pred.reshape(args.batch_size,16,16,16)
            voxels_src = voxels_gt.reshape(args.batch_size,32,32,32)
            print('voxels_src: ',voxels_src.shape)  
            try:      
                # pred = pytorch3d.ops.cubify(voxels_src,thresh = 0.5)
                vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
                vertices_src = torch.tensor(vertices_src).float()
                faces_src = torch.tensor(faces_src.astype(int))
                pred = pytorch3d.structures.Meshes([vertices_src], [faces_src]) 
                color=[0.7, 0.7, 1]   
                lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -2.0]], device=args.device)
                textures = (torch.ones_like(pred.verts_padded()).to(args.device))*(torch.tensor(color).to(args.device))
                pred.textures = pytorch3d.renderer.TexturesVertex(textures)
                renderer = get_mesh_renderer(image_size=256, device=args.device)
                rend = renderer(pred.to(args.device), cameras=cameras, lights=lights)
                rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
                plt.imsave(f'vis/{step}_implicit_{args.model_name}.png', rend)
            except:
                # pred = pytorch3d.ops.cubify(voxels_src,thresh = 0.3)
                vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
                vertices_src = torch.tensor(vertices_src).float()
                faces_src = torch.tensor(faces_src.astype(int))
                pred = pytorch3d.structures.Meshes([vertices_src], [faces_src]) 
                color=[0.7, 0.7, 1]   
                lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -2.0]], device=args.device)
                textures = (torch.ones_like(pred.verts_padded()).to(args.device))*(torch.tensor(color).to(args.device))
                pred.textures = pytorch3d.renderer.TexturesVertex(textures)
                renderer = get_mesh_renderer(image_size=256, device=args.device)
                rend = renderer(pred.to(args.device), cameras=cameras, lights=lights)
                rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
                plt.imsave(f'vis/{step}_implicit_{args.model_name}.png', rend)



      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['IoU@0.050000']
        avg_f1_score_05.append(f1_05)
        # avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        # avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["IoU@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); IoU@0.05: %.3f; Avg IoU@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
