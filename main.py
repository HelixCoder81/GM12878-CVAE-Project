import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# 假设你已经将之前的模块保存为 model.py, dataloader.py 和 engine.py
from src.model import GM12878_CVAE_Pro
from src.dataloader import GM12878CoolerDataset
from src.engine import train_one_epoch


def main(args):
    # 1. 环境准备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f">>> 使用设备: {device} | 任务: GM12878 3D 重建")

    # 2. 加载数据集 (使用实战级的 Cooler 加载器)
    print(f">>> 正在加载数据: {args.data_path}")
    train_dataset = GM12878CoolerDataset(
        cool_path=args.data_path,
        res=args.resolution,
        window_size=args.window_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 3. 初始化模型、优化器与缩放器 (AMP)
    model = GM12878_CVAE_Pro(
        input_size=args.window_size,
        latent_dim=args.latent_dim,
        label_dim=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()  # 混合精度，加速训练并省显存

    # 4. 核心训练循环
    print(">>> 开始训练...")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 调用之前写的 engine 中的训练步
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}] | Loss: {train_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 5. 定期保存权重
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)

        if epoch % 10 == 0:
            print(f">>> 已保存最优模型至: {checkpoint_path}")

    print(">>> 训练任务完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GM12878 CVAE 3D Chromatin Reconstruction")

    # 路径参数
    parser.add_argument("--data_path", type=str, required=True, help="Cooler文件路径 (.mcool)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="结果保存目录")

    # 超参数
    parser.add_argument("--resolution", type=int, default=10000, help="Hi-C 分辨率 (如 5000, 10000)")
    parser.add_argument("--window_size", type=int, default=64, help="切片窗口大小 (N x N)")
    parser.add_argument("--latent_dim", type=int, default=256, help="潜空间维度")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批大小")
    parser.add_argument("--epochs", type=int, default=100, help="总轮次")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda", help="指定设备 (cuda/cpu)")

    args = parser.parse_args()
    main(args)