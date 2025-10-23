import os
from jaxrl.agent.brc_learner import BRC
from jaxrl.envs import ParallelEnv
from jaxrl.env_names import get_environment_list
# import cv2
import argparse
import imageio.v3 as iio
from absl import app, flags


os.environ['MUJOCO_GL'] = 'egl'

episode_len = 900
flag = flags.FLAGS
flags.DEFINE_string('ckp', 'HB_NOHANDS', 'Name of the environment to use.')


def main(_):
    submit_dir = os.environ.get('SLURM_SUBMIT_DIR') if os.environ.get('SLURM_SUBMIT_DIR') is not None else '.'
    save_dir = r'/pfs/work9/workspace/scratch/ka_et4232-tcx'
    checkpoint_name = flag.ckp
    checkpoint_dir = f'{save_dir}/checkpoints/{checkpoint_name}'

    env_name = str(checkpoint_name)
    env_names = get_environment_list(env_name)
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
    agent.load_inference(checkpoint_dir)

    eval_stats = env.evaluate(agent, num_episodes=1, temperature=0.0, render=True, max_render_steps=episode_len)
    renders = eval_stats['renders']
    videos_dir = f'{submit_dir}/videos/{env_name}'
    os.makedirs(videos_dir, exist_ok=True)
    for i in range(renders.shape[0]):
        frames = renders[i]  # shape: (num_frames, channels, height, width)
        frames = frames.transpose(0, 2, 3, 1)  # Rearrange to (num_frames, height, width, channels)
        frames = (frames * 255).astype('uint8') if frames.dtype != 'uint8' else frames
        # frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]  # Convert RGB to BGR

        # height, width = frames[0].shape[:2]
        video_path = os.path.join(videos_dir, f'video_{i}.mp4')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        iio.imwrite(video_path, frames, fps=60)

    
if __name__ == "__main__":
    app.run(main)


