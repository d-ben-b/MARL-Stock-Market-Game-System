"""make_video.py — render a side-by-side comparison clip of two trained policies
driving the SAME scenes, to visualise the paper's central finding (similar
training reward, opposite deployed behaviour).

Top row = DQN (safe), bottom row = PPO (crashes). Both are evaluated under the
common base test config with identical reset seeds, so the traffic is the same
and only the policy differs.

Usage (from repo root):
    python src/make_video.py --env merge-v0 --episodes 4 --out logs/demo_merge.mp4
Then compress to a small file:
    ffmpeg -y -i logs/demo_merge.mp4 -vf scale=-2:480 -crf 30 -preset slow logs/demo_merge_small.mp4
"""
import os, sys, argparse
import numpy as np
import torch
import gymnasium as gym
import highway_env  # noqa: F401  (registers envs)
import imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from agents.networks import AttentionActorCritic, MlpActorCritic, MlpQNetwork
from envs.reward_shaper import get_env_config

DEVICE = torch.device("cpu")


def build_net(algo, arch, obs_shape, n_actions):
    if algo == "ppo":
        return (AttentionActorCritic(obs_shape, n_actions, 4) if arch == "attention"
                else MlpActorCritic(obs_shape, n_actions)).to(DEVICE)
    return MlpQNetwork(obs_shape, n_actions).to(DEVICE)


def load_policy(model_path, algo, arch, obs_shape, n_actions):
    net = build_net(algo, arch, obs_shape, n_actions)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()

    def act(obs):
        t = torch.Tensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if algo == "ppo":
                feat = (net.extract_features(t) if arch == "attention"
                        else net.feature_extractor(t))
                return torch.argmax(net.actor(feat), dim=1).item()
            return torch.argmax(net(t), dim=1).item()
    return act


def label_bar(width, text, color=(30, 30, 30)):
    bar = Image.new("RGB", (width, 26), color)
    d = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 4), text, fill=(255, 255, 255), font=font)
    return np.array(bar)


def tag_frame(frame, text, rgb):
    """Overlay a status word (e.g. CRASHED) on a frame."""
    img = Image.fromarray(frame.copy())
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    d.text((10, 6), text, fill=rgb, font=font)
    return np.array(img)


def run_pair(env_id, duration, top, bot, seeds):
    """top/bot = dict(model, algo, arch, label). Returns list of stacked frames."""
    cfg = get_env_config(env_id, "base", duration=duration)
    e_top = gym.make(env_id, render_mode="rgb_array"); e_top.unwrapped.configure(cfg)
    e_bot = gym.make(env_id, render_mode="rgb_array"); e_bot.unwrapped.configure(cfg)

    o, _ = e_top.reset(seed=0)
    act_top = load_policy(top["model"], top["algo"], top["arch"], o.shape, e_top.action_space.n)
    act_bot = load_policy(bot["model"], bot["algo"], bot["arch"], o.shape, e_bot.action_space.n)

    frames = []
    for sd in seeds:
        ot, _ = e_top.reset(seed=sd)
        ob, _ = e_bot.reset(seed=sd)
        dt = db = False
        last_t = e_top.render(); last_b = e_bot.render()
        crashed_t = crashed_b = False
        steps = 0
        while not (dt and db) and steps < duration * 5:
            if not dt:
                ot, _, term, trunc, it = e_top.step(act_top(ot))
                last_t = e_top.render()
                if it.get("crashed", False): crashed_t = True; dt = True
                dt = dt or term or trunc
            if not db:
                ob, _, term, trunc, ib = e_bot.step(act_bot(ob))
                last_b = e_bot.render()
                if ib.get("crashed", False): crashed_b = True; db = True
                db = db or term or trunc

            ft = tag_frame(last_t, "CRASHED", (255, 60, 60)) if crashed_t else last_t
            fb = tag_frame(last_b, "CRASHED", (255, 60, 60)) if crashed_b else last_b
            w = max(ft.shape[1], fb.shape[1])
            stack = np.vstack([
                label_bar(w, top["label"], (20, 90, 20)), ft,
                label_bar(w, bot["label"], (90, 20, 20)), fb,
            ])
            frames.append(stack)
            steps += 1
        # hold the final frame for ~1s so the outcome is readable
        for _ in range(5):
            frames.append(frames[-1])

    e_top.close(); e_bot.close()
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="merge-v0")
    ap.add_argument("--duration", type=int, default=80)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--out", default="logs/demo_merge.mp4")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    es = args.env.replace("-v0", "").replace("-", "_")
    TOP = dict(model=f"models/custom_model_dqn_mlp_aggressive_{es}_dur80_seed0.pth",
               algo="dqn", arch="mlp", label="DQN + MLP  (0% crash)")
    BOT = dict(model=f"models/custom_model_ppo_attention_aggressive_{es}_dur80_seed0.pth",
               algo="ppo", arch="attention", label="PPO + Attention  (100% crash)")

    print(f"Rendering {args.episodes} paired episodes on {args.env} ...")
    frames = run_pair(args.env, args.duration, TOP, BOT, seeds=list(range(args.episodes)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"Saved raw video: {args.out}  ({len(frames)} frames)")
