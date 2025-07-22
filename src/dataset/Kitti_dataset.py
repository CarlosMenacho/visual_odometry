import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from utils import rotation_to_euler
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import random


class KITTI(torch.utils.data.Dataset):
    """
    Dataloader for KITTI Visual Odometry Dataset
    http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    """

    def __init__(self,
                 data_path="/home/sergio/conferencia/TSformer-VO/kitti_data/data_odometry_gray/sequences",
                 gt_path="/home/sergio/conferencia/TSformer-VO/kitti_data/data_odometry_poses/poses",
                 camera_id="0",
                 sequences=["00", "02", "08", "09"],
                 window_size=3,
                 overlap=1,
                 read_poses=True,
                 transform=None):

        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.frame_id = 0
        self.read_poses = read_poses
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

        # KITTI normalization
        self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        self.sequences = sequences

        # Leer frames y ground truth
        frames, seqs = self.read_frames()
        gt = self.read_gt()

        data = pd.DataFrame({"gt": gt})
        data = data["gt"].apply(pd.Series)
        data["frames"] = frames
        data["sequence"] = seqs
        self.data = data
        self.windowed_data = self.create_windowed_dataframe(data)

    def __len__(self):
        return len(self.windowed_data["w_idx"].unique())

    def __getitem__(self, idx):
        data = self.windowed_data.loc[self.windowed_data["w_idx"] == idx, :]

        frames = data["frames"].values
        imgs = []
        for fname in frames:
            img = Image.open(fname).convert('RGB')
            img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.asarray(imgs)
        imgs = imgs.transpose(1, 0, 2, 3)  # TCHW → CTHW

        gt_poses = data.loc[:, [i for i in range(12)]].values
        y = []
        for gt_idx, gt in enumerate(gt_poses):
            pose = np.vstack([np.reshape(gt, (3, 4)), [[0., 0., 0., 1.]]])
            if gt_idx > 0:
                pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
                R = pose_wrt_prev[:3, :3]
                t = pose_wrt_prev[:3, 3]
                angles = rotation_to_euler(R, seq='zyx')
                angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
                t = (np.asarray(t) - self.mean_t) / self.std_t
                y.append(list(angles) + list(t))
            pose_prev = pose

        y = np.asarray(y).flatten()
        return imgs, y

    def read_intrinsics_param(self):
        calib_file = os.path.join(self.data_path, self.sequence, "calib.txt")
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            line = lines[int(self.camera_id)].strip().split()
            [fx, cx, fy, cy] = [float(line[1]), float(line[3]), float(line[6]), float(line[7])]
            self.cam_params["fx"] = fx
            self.cam_params["fy"] = fy
            self.cam_params["cx"] = cx
            self.cam_params["cy"] = cy

    def read_frames(self):
        frames = []
        seqs = []
        for sequence in self.sequences:
            frames_dir = os.path.join(self.data_path, sequence, f"image_{self.camera_id}", "*.png")
            frames_seq = sorted(glob.glob(frames_dir))
            frames += frames_seq
            seqs += [sequence] * len(frames_seq)
        return frames, seqs

    def read_gt(self):
        if self.read_poses:
            gt = []
            for sequence in self.sequences:
                path = os.path.join(self.gt_path, sequence + ".txt")
                with open(path) as f:
                    lines = f.readlines()
                    for line in lines:
                        line = [float(x) for x in line.strip().split()]
                        gt.append(line)
        else:
            gt = None
        return gt

    def create_windowed_dataframe(self, df):
        window_size = self.window_size
        overlap = self.overlap
        windowed_df = pd.DataFrame()
        w_idx = 0

        for sequence in df["sequence"].unique():
            seq_df = df[df["sequence"] == sequence].reset_index(drop=True)
            row_idx = 0
            while row_idx + window_size <= len(seq_df):
                rows = seq_df.iloc[row_idx:(row_idx + window_size)].copy()
                rows["w_idx"] = [w_idx] * len(rows)
                row_idx += window_size - overlap
                w_idx += 1
                windowed_df = pd.concat([windowed_df, rows], ignore_index=True)
        return windowed_df.reset_index(drop=True)
    
    def denormalize_pose(self, y):
        poses = []
        for i in range(self.window_size - 1):
            angles = y[i*6:i*6+3] * self.std_angles + self.mean_angles
            t = y[i*6+3:i*6+6] * self.std_t + self.mean_t
            poses.append((angles, t))
        return poses

if __name__ == "__main__":
    preprocess = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]),
    ])

    data = KITTI(
        transform=preprocess,
        sequences=["00"],
        window_size=3,
        overlap=2
    )

    # Selección aleatoria de una ventana
    random_idx = random.randint(0, len(data) - 1)
    imgs, gt = data[random_idx]

    # Imagen central
    center_img = imgs[:, imgs.shape[1] // 2]
    img_np = np.transpose(center_img, (1, 2, 0))
    img_np = (img_np * np.array([0.30737526, 0.31515116, 0.32020183])) + np.array([0.34721234, 0.36705238, 0.36066107])
    img_np = np.clip(img_np, 0, 1)

    # Denormalizar poses
    poses = data.denormalize_pose(gt)

    # Crear figura
    for i, (angles, t) in enumerate(poses):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(f"Ventana #{random_idx} - Paso {i+1}", fontsize=14)

        # Centro de imagen para dibujar flecha
        origin_x, origin_y = img_np.shape[1] // 2, img_np.shape[0] // 2
        scale = 100

        dx = t[0] * scale  # x
        dy = -t[2] * scale  # z

        # Dibujar flecha de movimiento
        ax.arrow(origin_x, origin_y, dx, dy,
                head_width=5, head_length=5, fc='lime', ec='lime', linewidth=2)

        # Información de pose (fijo en esquina superior izquierda)
        info = (
            f"Ángulos (Z, Y, X): [{angles[0]:.5f}, {angles[1]:.5f}, {angles[2]:.5f}]\n"
            f"Traslación [x, y, z]: [{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}]"
        )

        ax.text(10, 10, info,
                fontsize=8,
                color='white',
                va='top',
                bbox=dict(facecolor='black', alpha=0.6))

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    # Espera final para ver las imágenes
    input("Presiona Enter para cerrar...")

