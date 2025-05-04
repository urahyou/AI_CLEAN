import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import galaxy
import os

class GalaxyDataset(Dataset):
    def __init__(self, data_path, img_path, target_h5_path, shape, scale, transform=None):
        """
        初始化 GalaxyDataset 类，从 HDF5 文件加载预计算的目标数据。

        参数：
            data_path (str): 包含 UV 坐标和可见度数据的 HDF5 文件路径。
            img_path (str): 包含 Galaxy10 DECals 图像数据的 HDF5 文件路径。
            target_h5_path (str): 包含预计算目标数据的 HDF5 文件路径。
            min_noise (float): 最小噪声水平，默认值为 0.01。
            max_noise (float): 最大噪声水平，默认值为 0.1。
            transform (callable, optional): 可选的图像变换函数。
        """
        self.data_path = data_path
        self.img_path = img_path
        self.target_h5_path = target_h5_path
        self.shape = shape
        self.scale = scale
        self.transform = transform

        # 验证目标 HDF5 文件是否存在
        if not os.path.exists(target_h5_path):
            raise FileNotFoundError(f"目标 HDF5 文件未找到：{target_h5_path}。请先运行预处理脚本生成该文件。")

        # 加载 UV 坐标和可见度数据
        self.u_sparse, self.v_sparse, self.vis_re_sparse, self.vis_im_sparse, \
        self.u_dense, self.v_dense, self.vis_re_dense, self.vis_im_dense = galaxy.load_h5_uvvis(data_path)

        # 加载图像数据
        self.img_data, _ = galaxy.load_h5(img_path)

        # 加载预计算的 mask 和 sort_indices
        self.mask = np.load("./data/mask.npy")
        self.sort_indices = np.load("./data/sort_indices.npy")

        # 计算 uv_dense_scaled
        uv_dense = np.stack((self.u_dense.flatten(), self.v_dense.flatten()), axis=1)
        max_base = np.max(np.stack((self.u_sparse.flatten(), self.v_sparse.flatten()), axis=1))
        fourier_resolution = int(len(uv_dense) ** 0.5)
        self.uv_dense_scaled = np.rint((uv_dense + max_base) / max_base * (fourier_resolution - 1) / 2) / (fourier_resolution - 1) - 0.5

        # 从 HDF5 文件加载预计算的目标数据
        print(f"正在从 {target_h5_path} 加载目标数据...")
        with h5py.File(target_h5_path, 'r') as f:
            self.target_data = f['target_data'][:]
        print("目标数据加载完成。")

    def __len__(self):
        """返回数据集的长度（图像数量）。"""
        return self.img_data.shape[0]

    def __getitem__(self, idx):
        """
        返回指定索引的脏图像和目标图像。

        参数：
            idx (int): 数据样本的索引。

        返回：
            tuple: (dirty, target)，脏图像和目标图像。
        """
        # 随机选择噪声水平
        # noise_level = np.random.uniform(self.min_noise, self.max_noise)
        noise_level = np.random.gamma(shape=self.shape, scale=self.scale) 
        
        # 获取脏图像
        dirty = self.get_dirty_with_noise(idx, noise_level)

        # 从预计算数据中获取目标图像
        target = self.target_data[idx]

        # 转换为 PyTorch 张量并添加通道维度
        dirty = torch.tensor(dirty, dtype=torch.float32).unsqueeze(0)  # [1, 128, 128]
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)  # [1, 128, 128]

        # 应用变换（如果有）
        if self.transform:
            dirty = self.transform(dirty)
            target = self.transform(target)

        return dirty, target, noise_level

    def get_dirty_with_noise(self, idx, noise_level):
        """生成带有噪声的脏图像。"""
        vis_dense = np.stack((self.vis_re_dense[:, idx], self.vis_im_dense[:, idx]), axis=1)

        # 计算 vis_dense 的最大值以生成噪声
        max_vis = np.max(np.abs(vis_dense))

        # 生成噪声
        vis_noise = np.random.normal(0, max_vis * noise_level, vis_dense.shape)

        # 添加噪声
        vis_dense = vis_dense + vis_noise

        # 排序并重塑
        sorted_vis = vis_dense[self.sort_indices]
        reshaped_vis = sorted_vis.reshape((128, 128, 2), order='C')
        vis_dense = reshaped_vis.transpose((2, 0, 1))

        # 应用掩码
        vis_zf = vis_dense * self.mask

        # 生成图像并归一化
        image_dir = galaxy.to_img(vis_zf[0], vis_zf[1], self.uv_dense_scaled)
        image_dir = self.minmax(image_dir)
        
        return image_dir

    def minmax(self, img):
        """归一化图像。"""
        img = img * 100
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-12)
        return img

# 示例用法
if __name__ == "__main__":
    # 设置数据路径
    data_root = '/home/zhouxinghui/git/VIC-DDPM-modified/data'
    data_path = f'{data_root}/eht_grid_128FC_200im_Galaxy10_DECals_full.h5'
    img_path = f'{data_root}/Galaxy10_DECals.h5'
    target_h5_path = f'{data_root}/target_data.h5'

    # 创建数据集实例，指定噪声范围
    dataset = GalaxyDataset(data_path, img_path, target_h5_path, min_noise=0.01, max_noise=0.2)

    # 创建 DataLoader，支持并行加载
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # 获取一批数据并打印形状
    dirty, target = next(iter(dataloader))
    print(f"脏图像形状: {dirty.shape}")  # 预期: [4, 1, 128, 128]
    print(f"目标图像形状: {target.shape}")  # 预期: [4, 1, 128, 128]