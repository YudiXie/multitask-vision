# %%
from pathlib import Path
import pandas as pd
import numpy as np
from dataset import load_image
from train import IMN_transform

from matplotlib import pyplot as plt

# %%
# index_path = Path('/om2/user/yu_xie/data/tdw_images/tdw_image_dataset_large/index_img_1350132.csv')
index_path = Path('/Users/yudixie//data/tdw_image_dataset_small_multi_env_hdri/index_img_5898.csv')
dset_path = index_path.parent

# %%
dset_index = pd.read_csv(index_path, index_col=0)

# %%
means_stds = pd.read_csv(dset_path.joinpath('norm_column_mean_std.csv'), index_col=0).iloc[0]
headers = dset_path.joinpath('img_meta_headers.txt').read_text(encoding="utf-8").split("\n")

# %%
index = np.random.randint(len(dset_index))
# index = 883
image_idx = index
scene_n, wnid, model_n = dset_index.iloc[index]
img_path = dset_path.joinpath('images', scene_n, wnid, model_n, f"img_img_{image_idx:010d}.jpg")
img_meta_path = dset_path.joinpath('images', scene_n, wnid, model_n, f"img_{image_idx:010d}_info.csv")

img_meta = pd.read_csv(img_meta_path, names=headers).iloc[0]

img = load_image(img_path)
width, height = img.size

print(f"image_idx: {image_idx}")
print(f"scene: {scene_n}, wind: {wnid}")
print(f"model: {model_n}")
print(f'skybox: {img_meta["skybox_name"]}')
print(f'screen_x: {img_meta["ty"]:.2f}, screen_y: {img_meta["tz"]:.2f}')
print(f'screen_x_frac: {(img_meta["ty"] / width + 0.5):.2f}, screen_y_frac: {(img_meta["tz"] / height + 0.5):.2f}')
print(f'distance: {img_meta["neg_x"]:.2f}')
print(f'Eular angles: {img_meta["euler_1"]:.2f}, {img_meta["euler_2"]:.2f}, {img_meta["euler_3"]:.2f}')

fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(width // 2 + img_meta['ty'],
           height // 2 - img_meta['tz'],
           s=50, c='r', marker='x', label='positions')

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate_point(point, angles):
    # Angles in radians
    alpha, beta, gamma = np.radians(angles)

    # Rotation matrices for X, Y, Z
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))

    return np.dot(R, point)

def plot_rotated_airplane(angles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define basic airplane model (simple representation)
    # Body - line from tail to nose
    body_points = np.array([[0, 0, 0], [1, 0, 0]])
    # Wings - two lines perpendicular to the body
    wing_points = np.array([[0.5, 0.2, 0], [0.5, -0.2, 0]])
    # Tail - vertical line at the back
    tail_points = np.array([[0, 0, 0], [0, 0, 0.2]])

    # Function to plot airplane parts
    def plot_airplane_parts(points, color):
        for i in range(len(points) - 1):
            p0 = rotate_point(points[i], angles)
            p1 = rotate_point(points[i + 1], angles)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color)

    # Plot original airplane
    plot_airplane_parts(body_points, 'blue')
    plot_airplane_parts(wing_points, 'blue')
    plot_airplane_parts(tail_points, 'blue')

    # Setting plot limits and labels
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# Example usage
plot_rotated_airplane((img_meta['euler_1'], img_meta['euler_2'], img_meta['euler_3']))  # Angles in degrees


