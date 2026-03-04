import torch
import torch.nn as nn
def physics_loss(coords, target_hic, eps=1e-8):
    # 1. Hi-C 接触重构 (Log-distance relationship)
    # dist = (1 / contact_freq)^0.33
    dist_pred = torch.cdist(coords, coords, p=2)

    # 转换为接触概率模型 (基于聚合物物理模型)
    recon_hic = 1.0 / (dist_pred + 1.0)
    recon_loss = nn.functional.mse_loss(recon_hic, target_hic.squeeze(1))

    # 2. Excluded Volume (原子排斥力)
    # 当两个点距离 < Rmin 时，产生剧烈排斥损失
    r_min = 0.5
    too_close = r_min - dist_pred
    ev_loss = torch.mean(nn.functional.relu(too_close) ** 2)

    # 3. Continuity (链条连续性)
    # 模拟 DNA 骨架，相邻点距离应保持稳定
    neighbor_dist = torch.norm(coords[:, 1:, :] - coords[:, :-1, :], dim=-1)
    continuity_loss = nn.functional.mse_loss(neighbor_dist, torch.ones_like(neighbor_dist))

    return recon_loss, ev_loss, continuity_loss


def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    for hic, labels in loader:
        hic, labels = hic.to(device), labels.to(device)

        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            coords, mu, logvar = model(hic, labels)

            recon_l, ev_l, cont_l = physics_loss(coords, hic)
            kld_l = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # 权重调优：重构为主，物理为辅
            loss = recon_l * 10.0 + kld_l * 0.1 + ev_l * 5.0 + cont_l * 2.0

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()