import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from torch.utils.data import DataLoader, Dataset
from nets import NeRF_by_yenchen, get_embedder_by_yenchen, run_network_by_yenchen
from data_loading_by_yenchen import load_blender_data
from ray_utilis import get_rays, render_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def main():
    checkpoint_path = "./data/lego/pretrained_model.pth"
    checkpoint = torch.load(checkpoint_path)
    embed_fn, input_ch = get_embedder_by_yenchen(10, 0)
    embeddirs_fn, input_ch_views = get_embedder_by_yenchen(4, 0)
    model = NeRF_by_yenchen(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views, skips=[4], use_viewdirs=True).to(device)
    model.load_state_dict(checkpoint["model"])

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

    images_test = images[i_test]
    poses_test = poses[i_test]

    rendered_images = np.zeros((poses_test.shape[0], H, W, 3))

    for i in tqdm.tqdm(range(poses_test.shape[0])):
        rays_o, rays_dir = get_rays(H, W, K, poses_test[i])

        rendered_images[i] = render_image(rays_o=rays_o, rays_dir=rays_dir, 
            network_fn=model, network_query_fn=network_query_fn,
            near=near, far=far, N_samples=64)

    anim_fig, ax = plt.subplots()
    im = ax.imshow(rendered_images[0], animated=True)
    plt.box(False)
    plt.axis('off')
    def update(frame):
        im.set_array(rendered_images[frame])
        return im,
    
    anim = animation.FuncAnimation(anim_fig, update, frames=rendered_images.shape[0], interval=200, blit=True)

    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot(2,2,1)
    ax.imshow(images_test[0])
    ax.set_title("Original Image")
    plt.box(False)
    plt.axis('off')
    ax = fig.add_subplot(2,2,2)
    ax.imshow(rendered_images[0])
    ax.set_title("Image rendered")
    plt.box(False)
    plt.axis('off')

    ax = fig.add_subplot(2,2,3)
    ax.imshow(images_test[1])
    ax.set_title("Original Image")
    plt.box(False)
    plt.axis('off')
    ax = fig.add_subplot(2,2,4)
    ax.imshow(rendered_images[1])
    ax.set_title("Image rendered")
    plt.box(False)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()