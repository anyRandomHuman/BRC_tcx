import os
from jaxrl.agent.brc_learner import BRC
from jaxrl.envs import ParallelEnv
from jaxrl.env_names import get_environment_list
import cv2
import argparse

parser = argparse.ArgumentParser(description="A script to demonstrate run options in Python.")
parser.add_argument('--ckp', type=str, default='brc-HB_NOHANDS-0', help='Name of the environment to use.')

checkpoint_name = parser.parse_args().ckp
env_name = str(checkpoint_name).split('-')[1]
episode_len = 1000

submit_dir = os.environ.get('SLURM_SUBMIT_DIR') if os.environ.get('SLURM_SUBMIT_DIR') is not None else '.'

env_names = get_environment_list(env_name)
checkpoint_dir = f'{submit_dir}/checkpoints/{env_name}'
num_tasks = len(env_names)

env = ParallelEnv(env_names, seed=0)

kwargs = {}
kwargs['updates_per_step'] = 2
kwargs['width_critic'] = 4096

agent = BRC(
        0,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        **kwargs,
    )
agent.load(checkpoint_dir)

eval_stats = env.evaluate(agent, num_episodes=episode_len, temperature=0.0, render=True)
renders = eval_stats['renders']
videos_dir = f'{submit_dir}/videos/{env_name}'
os.makedirs(videos_dir, exist_ok=True)
for i in range(renders.shape[0]):
    frames = renders[i]  # shape: (num_frames, H, W, C)
    height, width = frames.shape[1], frames.shape[2]
    video_path = os.path.join(videos_dir, f'video_{i}.mp4')
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        out.write(frame_bgr)
    out.release()
