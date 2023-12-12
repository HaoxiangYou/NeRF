import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from nets import NeRF_by_yenchen, get_embedder_by_yenchen, run_network_by_yenchen
from data_loading_by_yenchen import load_blender_data
from ray_utilis import get_rays, render_rays, render_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# hyperparameter for training
N_batch = 256
N_epochs = 10
N_samples = 64
lr = 1e-3

class NeRFDataset(Dataset):
    def __init__(self, rays_o, rays_dir, view_dirs, rgbs):
        super().__init__()
        self.rays_o = rays_o.astype(np.float32)
        self.rays_dir = rays_dir.astype(np.float32)
        self.view_dirs = view_dirs.astype(np.float32)
        self.rgbs = rgbs.astype(np.float32)

    def __len__(self):
        return self.rays_o.shape[0]
    
    def __getitem__(self, index):
        return self.rays_o[index], self.rays_dir[index], self.view_dirs[index], self.rgbs[index]
    
def train(network_query_fn, network_fn, dataloader, optimizer, epochs, near, far, N_samples, logs_basedir):
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for i, (rays_o, rays_dir, view_dirs, rgbs) in enumerate(dataloader):
            z = torch.linspace(near, far, N_samples, dtype=torch.float32)
            Cs = render_rays(rays_o_batch=rays_o, rays_dir_batch=rays_dir,
                            view_dirs_batch=view_dirs, z=z, 
                            network_query_fn=network_query_fn, network_fn=network_fn)
            
            loss = loss_fn(Cs, rgbs)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"epoch: {epoch+1}, loss: {total_loss / len(dataloader)}")

def main():

    logs_basedir = "./logs/lego"

    embed_fn, input_ch = get_embedder_by_yenchen(10, 0)
    embeddirs_fn, input_ch_views = get_embedder_by_yenchen(4, 0)
    model = NeRF_by_yenchen(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views, skips=[4], use_viewdirs=True).to(device)

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network_by_yenchen(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=65536)
    
    images, poses, render_poses, hwf, i_split = load_blender_data("./data/lego", half_res=True, testskip=8)
    print('Loaded blender', images.shape, hwf)
    i_train, i_val, i_test = i_split

    near = 2.
    far = 6.
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])

    # setup intrinsics matrix
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
    ])

    # setup the training process
    images_train = images[i_train]
    poses_train = poses[i_train]

    images_test = images[i_test]
    poses_test = poses[i_test]

    rays_rgb_pair_train = np.array([get_rays(H, W, K, pose) + (image, ) for pose, image in zip(poses_train, images_train)])

    rays_o_train = rays_rgb_pair_train[:,0,...][0]
    rays_dir_train = rays_rgb_pair_train[:,1,...][0]
    view_dirs_train = rays_dir_train / np.linalg.norm(rays_dir_train, axis=-1, keepdims=True)[0]
    rgbs_train = rays_rgb_pair_train[:,2,...][0]

    nerf_dataset = NeRFDataset(rays_o_train.reshape((-1,3)), 
                               rays_dir_train.reshape((-1,3)), 
                               view_dirs_train.reshape((-1,3)), 
                               rgbs_train.reshape((-1,3)))

    nerf_dataloader = DataLoader(nerf_dataset, batch_size=N_batch, shuffle=True, drop_last=False,generator=torch.Generator(device=device))

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training the neural networks
    train(network_query_fn=network_query_fn, network_fn=model, 
        dataloader=nerf_dataloader, optimizer=optimizer, 
        epochs=N_epochs, N_samples=N_samples, near=near, far=far, 
        logs_basedir=logs_basedir)
    
    rays_o, rays_dir = get_rays(H, W, K, poses[0])

    rendered_image = render_image(rays_o=rays_o, rays_dir=rays_dir, 
        network_fn=model, network_query_fn=network_query_fn,
        near=near, far=far, N_samples=64)
    
    fig = plt.figure(figsize=(30,20))
    ax = fig.add_subplot(1,2,1)
    ax.imshow(images_train[0])
    ax.set_title("Original Image")
    plt.box(False)
    plt.axis('off')
    ax = fig.add_subplot(1,2,2)
    ax.imshow(rendered_image)
    ax.set_title("Image rendered")
    plt.box(False)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()