import argparse
import time

import dataset_location
import losses
import torch
from model_implicit import SingleViewto3D
# from model_implicit_PerceiverAdaLN import SingleViewto3D
# from model_implicit_OccNet import SingleViewto3D
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
# from r2n2_custom import R2N2
from dataset_creation.OccupancyGridDataLoader import OccupancyGridDataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def custom_collate_fn(batch):
    occupancy = torch.stack([item['occupancy'] for item in batch])
    view = torch.stack([item['view'] for item in batch])
    return {
        'voxels': occupancy,  # renaming to match your training script
        'images': view,
    }


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    # parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max_iter", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh","implicit"], type=str
    )
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    # parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--device', default='cuda:0', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument("--num_samples", default=4096, type=int)
    # parser.add_argument("--num_samples", default=32768, type=int)
    parser.add_argument("--model_name", default="OccNet", type=str)
    return parser


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    # num_samples = 1000 
    # num_samples = 32*32*32
    indices = torch.randperm(64*64*64)[:args.num_samples]  
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    elif args.type == "implicit":
        ground_truth_3d = feed_dict["voxels"].float()

        ground_truth_3d = ground_truth_3d.reshape(args.batch_size,-1)[:,indices]  
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device), indices


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    elif args.type == "implicit":
        loss = losses.voxel_loss(predictions, ground_truth)
    return loss


def get_midpt_and_range(lims):
    midpt = (lims[1] + lims[0])/2.
    span  = abs(lims[1] - lims[0])
    return midpt, span

def equal_axes(ax):
    x_m, x_r = get_midpt_and_range(ax.get_xlim3d())
    y_m, y_r = get_midpt_and_range(ax.get_ylim3d())
    z_m, z_r = get_midpt_and_range(ax.get_zlim3d())

    r = max([x_r, y_r, z_r])/2.
    ax.set_xlim3d([x_m - r, x_m + r])
    ax.set_ylim3d([y_m - r, y_m + r])
    ax.set_zlim3d([z_m - r, z_m + r])


def plot_field(xyz, field, cmap="coolwarm", title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=field, cmap=cmap)
    equal_axes(ax)
    plt.title(title)
    #ax.view_init(10,60)
    plt.show()

def plot_fields(xyz, field, titles = None, cmap="coolwarm"):

    # vmin = min([min(f) for f in fields])
    # vmax = max([max(f) for f in fields])

    # fig = plt.figure(figsize=(3.5*N,4),dpi=300)
    # fig = plt.figure(figsize=(3.5*N,4))
    # for i, field in enumerate(fields):
    #     ax = fig.add_subplot(1, N, i+1, projection='3d')
    #     scatter = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=field, cmap=cmap, vmin = vmin, vmax = vmax)
    #     # ax.set_xticklabels([])
    #     # ax.set_yticklabels([])
    #     # ax.set_zticklabels([])
    #     equal_axes(ax)
    #     plt.title(titles[i])
    #     cbar = fig.colorbar(scatter, shrink=0.8)

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # scatter = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=field, cmap=cmap)
    scatter = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=field, s=field*100, alpha = 0.3, cmap=cmap)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # equal_axes(ax)
    # plt.title(titles[i])
    ax.view_init(30, 30,vertical_axis='y')
    cbar = fig.colorbar(scatter)
    plt.savefig('vox.png')

def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train_model(args):
    print(dataset_location.FUSION_360_OCCUPANCY_PATH)
    print(dataset_location.FUSION_360_VIEWS_PATH)
    print(dataset_location.SPLITS_PATH)
    # r2n2_dataset = R2N2(
    #     "train",
    #     dataset_location.SHAPENET_PATH,
    #     dataset_location.R2N2_PATH,
    #     dataset_location.SPLITS_PATH,
    #     return_voxels=True,
    #     return_feats=args.load_feat,
    # )

    f360_dataset = OccupancyGridDataset(
        occupancy_dir=dataset_location.FUSION_360_OCCUPANCY_PATH,
        image_dir=dataset_location.FUSION_360_VIEWS_PATH,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]),
        split_path = dataset_location.SPLITS_PATH,
        split_name='train'
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
    train_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()
    params = count_parameters(model)


    # # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()
    print('args.load_checkpoint',args.load_checkpoint)
    if args.load_checkpoint:
        print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")

    print("Starting training !")
    print('len(train_loader): ',len(train_loader))
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)
        print(feed_dict.keys())

        images_gt, ground_truth_3d, indices = preprocess(feed_dict, args)
        print(ground_truth_3d.shape)
        read_time = time.time() - read_start_time

        print('images_gt',images_gt.shape)
        print('ground_truth_3d',ground_truth_3d.shape)
        print('ground_truth_3d.reshape(args.batch_size,-1,1)',ground_truth_3d.reshape(args.batch_size,-1,1).shape)
        ground_truth_3d = ground_truth_3d.reshape(args.batch_size,-1)

        xl = np.linspace(-1,1,64)
        yl = np.linspace(-1,1,64)  
        zl = np.linspace(-1,1,64)
        xc, yc, zc = np.meshgrid(xl, yl, zl, indexing='ij')
        coords = np.concatenate((xc.reshape(-1,1),yc.reshape(-1,1),zc.reshape(-1,1)),axis=1)
        print('coords[indices,:] ',coords[indices,:].shape)
        coords = torch.tensor(coords, dtype = torch.float32, device = args.device)
    #     # plot_fields(coords[indices,:],ground_truth_3d.reshape(args.batch_size,-1,1)[0,:,:].cpu().detach().numpy())

        prediction_3d = model(images_gt, args, indices)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0 and step > 0 or step == args.max_iter:
            print(f"Saving checkpoint at step {step}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoint_{args.type}_{args.model_name}.pth",
            )

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
