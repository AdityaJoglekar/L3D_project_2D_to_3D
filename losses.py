import torch
import pytorch3d
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	loss_func = torch.nn.BCELoss()
	# print('target: ',torch.max(voxel_tgt), torch.min(voxel_tgt))
	voxel_src = torch.clamp(voxel_src, min=1e-4, max=1-1e-4)
	# print('voxel_tgt',voxel_tgt.shape)
	loss = loss_func(voxel_src,voxel_tgt)
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 

	# y ref, x query
	knn= pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1)
	# knn= pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1, return_nn = True)
	# print('knn',knn[2].squeeze(2))
	# gathered_points = pytorch3d.ops.knn_gather(point_cloud_tgt, knn[1])
	# gathered_points = knn[2].squeeze(2)
	# loss1 = torch.mean((point_cloud_src - gathered_points)**2)
	loss1 = torch.mean(knn.dists[:,:,0])

	# x ref, y query
	knn = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1)
	# knn = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1, return_nn = True)
	# gathered_points = pytorch3d.ops.knn_gather(point_cloud_src, knn[1])
	# gathered_points = knn[2].squeeze(2)
	# loss2 = torch.mean((point_cloud_tgt - gathered_points)**2)	
	loss2 = torch.mean(knn.dists[:,:,0])

	loss_chamfer = loss1 + loss2
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	loss_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian