"""
采集演示数据：用预设关键帧插值生成"机械臂移动到电池包上方"的演示轨迹。
保存为 npz，后续用 Diffusion Policy 模仿学习。

用法：
    python collect_demo.py --num_episodes 50 --output data/
"""
import argparse
import numpy as np
import mujoco
import os

# 预设关键帧：6 个关节的角度（弧度）
# 顺序：j1(腰转) j2(大臂) j3(小臂) j4(腕转) j5(腕俯仰) j6(夹爪)
KEYFRAMES = np.array([
    [0.0, -0.5, 0.5, 0.0, 0.0, 0.0],   # 0: home 位置，手臂收在身前
    [0.0, -1.0, 0.8, 0.0, 0.3, 0.0],   # 1: 抬起大臂
    [0.6, -1.2, 1.0, 0.0, 0.5, 0.0],   # 2: 转向电池包方向
    [0.8, -0.8, 0.6, 0.0, 0.3, 0.0],   # 3: 伸出去，向电池包靠近
    [0.85, -0.6, 0.4, 0.0, 0.2, 0.0],  # 4: 到达电池包上方
    [0.85, -0.6, 0.4, 0.0, 0.2, 0.0],  # 5: 保持
])

def interpolate_trajectory(keyframes, steps_per_kf=20):
    """关键帧之间线性插值，生成平滑轨迹。"""
    traj = []
    for i in range(len(keyframes) - 1):
        for s in range(steps_per_kf):
            alpha = s / steps_per_kf
            traj.append((1 - alpha) * keyframes[i] + alpha * keyframes[i + 1])
    traj.append(keyframes[-1])
    return np.array(traj)

def collect_episode(model, data, target_joints, noise_std=0.02):
    """跑一条 episode：按 target_joints 控制机械臂，记录观测和动作。"""
    obs_list = []
    act_list = []

    # 重置到 home
    data.qpos[:6] = KEYFRAMES[0]
    mujoco.mj_forward(model, data)

    for target_q in target_joints:
        # 加一点噪声，让数据不那么完美
        noisy_target = target_q + np.random.normal(0, noise_std, size=6)
        # PD 控制：用 position control（mujoco 内部会算力矩）
        for _ in range(5):  # 每个目标步跑 5 个仿真步
            data.ctrl[:6] = noisy_target
            mujoco.mj_step(model, data)
        # 记录当前观测（关节角度）和动作（目标关节角度）
        obs = data.qpos[:6].copy()
        ee_pos = data.site_xpos[model.site('ee_site').id].copy()
        obs_list.append(np.concatenate([obs, ee_pos]))
        act_list.append(noisy_target)

    return np.array(obs_list), np.array(act_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--output', type=str, default='data/')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    model = mujoco.MjModel.from_xml_path(os.path.join(here, 'scene.xml'))
    data = mujoco.MjData(model)

    os.makedirs(args.output, exist_ok=True)

    all_obs = []
    all_act = []
    traj = interpolate_trajectory(KEYFRAMES, steps_per_kf=20)
    print(f"演示轨迹长度：{len(traj)} 步")

    for ep in range(args.num_episodes):
        obs, act = collect_episode(model, data, traj)
        all_obs.append(obs)
        all_act.append(act)
        if (ep + 1) % 10 == 0:
            print(f"  采集 {ep+1}/{args.num_episodes}")

    all_obs = np.array(all_obs)  # (N, T, obs_dim)
    all_act = np.array(all_act)  # (N, T, act_dim)
    save_path = os.path.join(args.output, 'auto_repair_demo.npz')
    np.savez(save_path, obs=all_obs, act=all_act)
    print(f"完成！保存到 {save_path}")
    print(f"  episodes: {all_obs.shape[0]}, T: {all_obs.shape[1]}, obs_dim: {all_obs.shape[2]}, act_dim: {all_act.shape[2]}")

if __name__ == '__main__':
    main()
