import torch
import numpy as np
import os
from src.model import GM12878_CVAE_Pro
from src.utils import export_to_pdb, kabsch_alignment


class ChromatinInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 1. 初始化并加载模型
        self.model = GM12878_CVAE_Pro(input_size=64, latent_dim=256).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"成功加载模型权重: {checkpoint_path}")

    @torch.no_grad()
    def predict_ensemble(self, hic_matrix, labels, num_samples=20):
        """
        生成构象系 (Ensemble): 模拟染色质在核内的动态性
        """
        hic_tensor = torch.from_numpy(hic_matrix).unsqueeze(0).unsqueeze(0).float().to(self.device)
        label_tensor = torch.from_numpy(labels).unsqueeze(0).float().to(self.device)

        ensemble = []
        for i in range(num_samples):
            # 通过 CVAE 的潜空间采样生成不同的可能结构
            coords, _, _ = self.model(hic_tensor, label_tensor)
            ensemble.append(coords.squeeze().cpu().numpy())

        return np.array(ensemble)

    def get_consensus_structure(self, ensemble):
        """
        使用 Kabsch 算法对齐所有生成的结构，并计算平均共识结构
        """
        ref_struct = ensemble[0]
        aligned_ensemble = [ref_struct]

        for i in range(1, len(ensemble)):
            # 对齐当前结构到参考结构
            aligned = kabsch_alignment(ensemble[i], ref_struct)
            aligned_ensemble.append(aligned)

        # 计算平均坐标
        consensus = np.mean(aligned_ensemble, axis=0)
        return consensus


# 使用示例
if __name__ == "__main__":
    # 模拟输入数据 (实际应从 DataLoader 或 Cooler 获取)
    mock_hic = np.random.rand(64, 64)
    mock_labels = np.array([0.8, 0.5, 0.2])  # TAD, AB, Loop 特征

    infer = ChromatinInference(checkpoint_path="outputs/best_model.pt")

    # 1. 生成 50 个可能的构象
    ensemble = infer.predict_ensemble(mock_hic, mock_labels, num_samples=50)

    # 2. 计算唯一的共识结构
    consensus_struct = infer.get_consensus_structure(ensemble)

    # 3. 导出为 PDB 文件
    os.makedirs("results", exist_ok=True)
    export_to_pdb(consensus_struct, "results/consensus_structure.pdb")
    print(">>> 推理完成！共识结构已保存至 results/consensus_structure.pdb")