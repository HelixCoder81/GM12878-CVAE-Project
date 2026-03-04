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