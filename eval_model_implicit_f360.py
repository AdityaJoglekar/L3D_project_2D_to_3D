import argparse
import time
import torch
# from model import SingleViewto3D
# from model_implicit_eval import SingleViewto3D
# from model_implicit_PerceiverAdaLN import SingleViewto3D
# from model_implicit_OccNet import SingleViewto3D
from model_implicit_OccNet_eval import SingleViewto3D
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
    parser.add_argument("--model_name", default="OccNet", type=str)
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)
    size = 64
    step = 2
    indices = torch.arange(0, size, step)  # tensor([0, 2, 4, ..., 62])
    images = feed_dict['images'].squeeze(1)
    voxels_gt = feed_dict['voxels'].float()
    voxels_gt = voxels_gt.reshape(size,size,size)[indices][:, indices][:, :, indices]
    voxels_gt = voxels_gt.reshape(args.batch_size,1,32,32,32)
    H,W,D = voxels_gt.shape[2:]
    vertices_src, faces_src = mcubes.marching_cubes(voxels_gt.detach().cpu().squeeze().numpy(), isovalue=0.5)
    # print('voxels_src',len(voxels_gt.detach().cpu().numpy()[np.where(voxels_gt.detach().cpu().numpy()>0.5)]))
    vertices_src = torch.tensor(vertices_src).float()
    faces_src = torch.tensor(faces_src.astype(int))
    mesh_gt = pytorch3d.structures.Meshes([vertices_src], [faces_src])  
    return images.to(args.device), mesh_gt, indices


def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    # plt.savefig(f'eval_{args.type}', bbox_inches='tight')
    plt.savefig(f'eval_{args.type}_{args.model_name}_implicit', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions.reshape(args.batch_size,1,32,32,32)
        # print('voxels_src',voxels_src.detach().cpu().numpy()[np.where(voxels_src.detach().cpu().numpy()>0.5)])
        # print('voxels_src',len(voxels_src.detach().cpu().numpy()[np.where(voxels_src.detach().cpu().numpy()>0.5)]))
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        # if vertices_src.shape == torch.Size([0, 3]):
            # print('here')
            # print('vertices',vertices_src[np.where(vertices_src>-1000)])
            # vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.2)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src]) 
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    # elif args.type == "point":
    #     pred_points = predictions.cpu()
    # elif args.type == "mesh":
    #     pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



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
        checkpoint = torch.load(f'checkpoint_implicit_{args.model_name}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt, indices = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        # predictions = model(images_gt, args, indices = torch.arange(0,64*64*64))
        predictions = model(images_gt, args, indices = indices)
        # print(predictions.shape)
        # print('pred',torch.max(predictions))

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        # TODO:
        # if (step % args.vis_freq) == 0:
        #     # visualization block
        #     #  rend = 
        #     plt.imsave(f'vis/{step}_{args.type}.png', rend)
        if (step % args.vis_freq) == 0:
            R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=200, elev=30, azim=120)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=args.device)

            color=[0.7, 0.7, 1]   
            lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -2.0]], device=args.device)
            textures = (torch.ones_like(mesh_gt.verts_padded()).to(args.device))*(torch.tensor(color).to(args.device))
            mesh_gt.textures = pytorch3d.renderer.TexturesVertex(textures)
            renderer = get_mesh_renderer(image_size=256, device=args.device)
            rend = renderer(mesh_gt.to(args.device), cameras=cameras, lights=lights)
            rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
            plt.imsave(f'vis/{step}_meshgt.png', rend)   
        
            pred = predictions
            voxels_src = pred.reshape(args.batch_size,32,32,32)
            print('voxels_src: ',voxels_src.shape) 

            try:      
                pred = pytorch3d.ops.cubify(voxels_src,thresh = 0.5)
                color=[0.7, 0.7, 1]   
                lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -2.0]], device=args.device)
                textures = (torch.ones_like(pred.verts_padded()).to(args.device))*(torch.tensor(color).to(args.device))
                pred.textures = pytorch3d.renderer.TexturesVertex(textures)
                renderer = get_mesh_renderer(image_size=256, device=args.device)
                rend = renderer(pred.to(args.device), cameras=cameras, lights=lights)
                rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
                plt.imsave(f'vis/{step}_implicit_{args.model_name}.png', rend)
            except:
                pred = pytorch3d.ops.cubify(voxels_src,thresh = 0.3)
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

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
