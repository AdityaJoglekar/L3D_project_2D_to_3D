import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch

import pytorch3d
from utils import get_device, get_mesh_renderer, get_points_renderer
from PIL import Image
import imageio
import numpy as np

def render_point_clouds(points_pred, points_gt):
    color=[0.7, 0.7, 1]
    device = get_device()
    
    point_colors = torch.ones_like(points_pred.squeeze(0)) * (torch.tensor(color).to(device))
    point_cloud_pred = pytorch3d.structures.Pointclouds(
        points=points_pred, features=[point_colors],
    ).to(device)

    point_colors = torch.ones_like(points_gt.squeeze(0))* (torch.tensor(color).to(device))
    point_cloud_gt = pytorch3d.structures.Pointclouds(
        points=points_gt, features=[point_colors],
    ).to(device)

    dists = torch.linspace(2, 2, 1)
    elevs = torch.linspace(-90, 90, 10)
    azims = torch.linspace(0, 350, 11)
    renders = []
    for dist in dists:
        for elev in elevs:
            for azim in azims:
                R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim)
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=R, T=T, fov=60, device=device
                )
                points_renderer = get_points_renderer(image_size=256, device=device)
                rend = points_renderer(point_cloud_pred, cameras=cameras)
                rend = rend[0, ..., :3].detach().cpu().numpy()
                renders.append(rend)
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave('outputs/p12_pred.gif', images, loop = 0)     

    renders = []
    for dist in dists:
        for elev in elevs:
            for azim in azims:
                R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim)
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=R, T=T, fov=60, device=device
                )
                points_renderer = get_points_renderer(image_size=256, device=device)
                rend = points_renderer(point_cloud_gt, cameras=cameras)
                rend = rend[0, ..., :3].detach().cpu().numpy()
                renders.append(rend)
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave('outputs/p12_gt.gif', images, loop = 0)  


def render_meshes(mesh_pred, mesh_gt):
    device = get_device()
    color=[0.7, 0.7, 1]
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -2.0]], device=device)

    textures = torch.ones_like(mesh_pred.verts_padded())*(torch.tensor(color).to(device))
    mesh_pred.textures = pytorch3d.renderer.TexturesVertex(textures)

    textures = torch.ones_like(mesh_gt.verts_padded())*(torch.tensor(color).to(device))
    mesh_gt.textures = pytorch3d.renderer.TexturesVertex(textures)

    dists = torch.linspace(2, 2, 1)
    elevs = torch.linspace(-90, 90, 10)
    azims = torch.linspace(0, 350, 11)

    renders = []
    for dist in dists:
        for elev in elevs:
            for azim in azims:
                R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim)
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=R, T=T, fov=60, device=device
                )
                renderer = get_mesh_renderer(image_size=256, device=device)
                rend = renderer(mesh_pred, cameras=cameras, lights=lights)
                rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
                renders.append(rend)
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave('outputs/p13_pred.gif', images, loop = 0)

    renders = []
    for dist in dists:
        for elev in elevs:
            for azim in azims:
                R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim)
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=R, T=T, fov=60, device=device
                )
                renderer = get_mesh_renderer(image_size=256, device=device)
                rend = renderer(mesh_gt, cameras=cameras, lights=lights)
                rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
                renders.append(rend)
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave('outputs/p13_gt.gif', images, loop = 0)    

def render_voxel(voxel_pred, voxel_gt):
    mesh_pred = pytorch3d.ops.cubify(voxel_pred,thresh = 0.5)
    mesh_gt = pytorch3d.ops.cubify(voxel_gt,thresh = 0.5)

    device = get_device()
    color=[0.7, 0.7, 1]
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -2.0]], device=device)

    textures = (torch.ones_like(mesh_pred.verts_padded()).to(args.device))*(torch.tensor(color).to(args.device))
    mesh_pred.textures = pytorch3d.renderer.TexturesVertex(textures)

    textures = (torch.ones_like(mesh_gt.verts_padded()).to(args.device))*(torch.tensor(color).to(args.device))
    mesh_gt.textures = pytorch3d.renderer.TexturesVertex(textures)
    dists = torch.linspace(2, 2, 1)
    elevs = torch.linspace(-90, 90, 10)
    azims = torch.linspace(0, 350, 11)

    renders = []
    for dist in dists:
        for elev in elevs:
            for azim in azims:
                R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim)
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=R, T=T, fov=60, device=device
                )
                renderer = get_mesh_renderer(image_size=256, device=device)
                rend = renderer(mesh_pred.to(args.device), cameras=cameras, lights=lights)
                rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
                renders.append(rend)
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave('outputs/p11_pred.gif', images, loop = 0)

    renders = []
    for dist in dists:
        for elev in elevs:
            for azim in azims:
                R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim)
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=R, T=T, fov=60, device=device
                )
                renderer = get_mesh_renderer(image_size=256, device=device)
                rend = renderer(mesh_gt.to(args.device), cameras=cameras, lights=lights)
                rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
                renders.append(rend)
    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave('outputs/p11_gt.gif', images, loop = 0)   

def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')
    render_meshes(mesh_src,mesh_tgt)


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    render_point_clouds(pointclouds_src, pointclouds_tgt)


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    render_voxel(voxels_src, voxels_tgt)

def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)        


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
