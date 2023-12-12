import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def get_rays(H, W, K, c2w):
    """
    Generate rays that project to each pixel of the image

    Args:
        H: height of image
        W: weight of image
        K: intristic matrix
        c2w: the transformation between camera frame and world frame
    Returns:
        rays_dir: (H X W X 3) the rays direction corresponding to each pixel
        rays_o: (H X W X 3) the rays origin (camera orign) corresponding to each pixel
    """
    K_inv = np.linalg.inv(K)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    i = i.flatten()
    j = j.flatten()
    ij_h = np.stack((i,j, np.ones_like(i)), axis=0)

    # convert the rays dir from image coordinates to the world frame
    # a change of coordinates was applied to deal with strange image cooridnates convention, where y point down
    rays_dir = c2w[:3,:3] @ np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32) @ K_inv @ ij_h
    rays_o = np.broadcast_to(c2w[:3, -1][:, None], rays_dir.shape)

    rays_dir = (rays_dir.T).reshape((H, W, 3))
    rays_o = (rays_o.T).reshape((H, W, 3))

    return rays_o.astype(np.float32), rays_dir.astype(np.float32)

def render_image(rays_o, rays_dir, network_fn, network_query_fn, near, far, N_samples, chunk=32768):

    image_shape = rays_dir.shape[:-1]
    rays_o_flatten = rays_o.reshape((-1,3))
    rays_dir_faltten = rays_dir.reshape((-1, 3))

    # the view direction is the same as the rays direction, 
    # however, the input to the neural should be a uniform dir
    view_dirs_flatten = (rays_dir_faltten / np.linalg.norm(rays_dir_faltten, axis=-1, keepdims=True)).astype(np.float32)

    # the distance along the rays
    z = torch.linspace(near, far, N_samples, dtype=torch.float32)
    
    rgb_image_flatten = np.zeros_like(rays_o_flatten)

    for i in range(0, rays_o_flatten.shape[0], chunk):
        # create a small batch of rays
        rays_o_batch = torch.tensor(rays_o_flatten[i:i+chunk], dtype=torch.float32).to(device)
        rays_dir_batch = torch.tensor(rays_dir_faltten[i:i+chunk], dtype=torch.float32).to(device)
        view_dirs_batch = torch.tensor(view_dirs_flatten[i:i+chunk]).to(device)

        # points in N_array
        pts = rays_o_batch[:, None, :] + rays_dir_batch[:, None, :] * z[None,:, None]
        
        with torch.no_grad():
        # get the raw data from neural networks
            raw = network_query_fn(pts, view_dirs_batch, network_fn)
            # get the rgb values from 0 to 1
            rgbs_volume = F.sigmoid(raw[...,:3])
            # get the volume density value to be nonnegative
            volume_densities = F.relu(raw[..., -1])
            distances = torch.linalg.norm(rays_dir_batch, dim=-1, keepdims=True) * (torch.concatenate([z[1:] - z[:-1], torch.tensor([1e10])])[None, :])
            Ts = torch.exp(-torch.cumsum(
                torch.concatenate([torch.zeros((volume_densities.shape[0],1)), (volume_densities * distances)[:,:-1]], dim=1), 
            dim=1))
            Cs = torch.sum( (Ts * (1 - torch.exp(-distances * volume_densities)))[..., None] * rgbs_volume, dim=1)
            
            rgb_image_flatten[i:i+chunk] = Cs.cpu().numpy()

    rgb_image = rgb_image_flatten.reshape(image_shape + (3,))

    return rgb_image

def test():
    """
    This function will going to verify the correctness of ray rendering function,
    To do so, we utilize the a pretrained nerf model by yenchen
    """

    # Load a pretrained model by yenchen
    ckpt_path = "data/lego/200000.tar"
    ckpt = torch.load(ckpt_path)

    embed_fn, input_ch = get_embedder_by_yenchen(10, 0)
    embeddirs_fn, input_ch_views = get_embedder_by_yenchen(4, 0)

    model = NeRF_by_yenchen(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views, skips=[4], use_viewdirs=True).to(device)
    model_fine = NeRF_by_yenchen(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views, skips=[4], use_viewdirs=True).to(device)

    model.load_state_dict(ckpt["network_fn_state_dict"])
    model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network_by_yenchen(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=65536)
    
    # Load a lego blender dataset by yenchen
    images, poses, render_poses, hwf, i_split = load_blender_data("./data/lego", half_res=True, testskip=8)
    print('Loaded blender', images.shape, hwf)
    i_train, i_val, i_test = i_split

    # The z distance that the ray passed
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

    image = images[0]
    pose = poses[0]

    rays_o, rays_dir = get_rays(H, W, K, pose)
    rendered_image = render_image(rays_o=rays_o, rays_dir=rays_dir, 
                network_fn=model_fine, network_query_fn=network_query_fn,
                near=near, far=far, N_samples=64)
    
    fig = plt.figure(figsize=(30,20))
    ax = fig.add_subplot(1,2,1)
    ax.imshow(image)
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
    from nets import NeRF_by_yenchen, get_embedder_by_yenchen, run_network_by_yenchen
    from data_loading_by_yenchen import load_blender_data
    test()