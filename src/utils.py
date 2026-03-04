import cooler
import numpy as np
from torch.utils.data import Dataset


class GM12878CoolerDataset(Dataset):
    def __init__(self, cool_path, res=10000, window_size=64):
        self.clr = cooler.Cooler(f"{cool_path}::resolutions/{res}")
        self.chroms = self.clr.chromnames
        self.window_size = window_size
        self.data_indices = self._make_index()

    def _make_index(self):
        # 扫描染色体，建立可用的滑动窗口索引
        indices = []
        for chrom in self.chroms:
            size = self.clr.chromsizes[chrom]
            for start in range(0, size // 10000 - self.window_size, self.window_size):
                indices.append((chrom, start))
        return indices

    def __getitem__(self, idx):
        chrom, start = self.data_indices[idx]
        # 获取平衡后的矩阵 (ICE normalization)
        mat = self.clr.matrix(balance=True).fetch(f"{chrom}:{start * 10000}-{(start + 64) * 10000}")
        mat = np.nan_to_num(mat)

        # 模拟对应的标签数据 (TAD, Loop)
        # 实际开发中需从 bedpe 文件匹配
        label = np.array([np.mean(mat), np.std(mat), 0.5], dtype=np.float32)

        return torch.from_numpy(mat).unsqueeze(0).float(), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_indices)


import numpy as np


def kabsch_alignment(P, Q):
    """
    Kabsch 算法：计算最优旋转矩阵，将 P 对齐到 Q
    """
    # 1. 去中心化
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)

    # 2. 计算协方差矩阵
    C = np.dot(P_centered.T, Q_centered)

    # 3. 奇异值分解 (SVD)
    V, S, Wt = np.linalg.svd(C)

    # 4. 计算旋转矩阵 (处理镜像情况)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, Wt)

    # 5. 返回旋转并平移后的坐标
    return np.dot(P_centered, U) + np.mean(Q, axis=0)


def export_to_pdb(coords, filename):
    """导出标准 PDB 格式"""
    with open(filename, 'w') as f:
        for i, (x, y, z) in enumerate(coords):
            f.write(f"ATOM  {i + 1:>5}  CA  MET A{i + 1:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00\n")
        # 写入骨架连接
        for i in range(1, len(coords)):
            f.write(f"CONECT{i:>5}{i + 1:>5}\n")