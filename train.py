"""
最简行为克隆（BC）训练：从演示数据学 obs -> act。
先跑通管线，后面再升级到 Diffusion Policy。

用法：
    python train.py --data data/auto_repair_demo.npz --epochs 200
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import os

class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/auto_repair_demo.npz')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    d = np.load(args.data)
    obs, act = d['obs'], d['act']
    # obs: (N, T, obs_dim), act: (N, T, act_dim) -> 展平成 (N*T, dim)
    N, T, obs_dim = obs.shape
    act_dim = act.shape[2]
    obs = obs.reshape(-1, obs_dim)
    act = act.reshape(-1, act_dim)
    print(f"数据：{N} episodes, {T} 步, obs_dim={obs_dim}, act_dim={act_dim}, 总样本={len(obs)}")

    # 标准化
    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-6
    act_mean, act_std = act.mean(0), act.std(0) + 1e-6
    obs_n = (obs - obs_mean) / obs_std
    act_n = (act - act_mean) / act_std

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备：{device}")

    model = MLP(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    obs_t = torch.tensor(obs_n, dtype=torch.float32).to(device)
    act_t = torch.tensor(act_n, dtype=torch.float32).to(device)

    for ep in range(args.epochs):
        idx = torch.randint(0, len(obs_t), (args.batch_size,))
        x, y = obs_t[idx], act_t[idx]
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            print(f"epoch {ep+1}/{args.epochs}  loss={loss.item():.6f}")

    # 保存
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'obs_mean': obs_mean, 'obs_std': obs_std,
        'act_mean': act_mean, 'act_std': act_std,
        'obs_dim': obs_dim, 'act_dim': act_dim,
    }, 'checkpoints/bc_mlp.pt')
    print("保存到 checkpoints/bc_mlp.pt")

if __name__ == '__main__':
    main()
