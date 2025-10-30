import os

import numpy as np

from jaxrl.agent.brc_learner import BRC
from jaxrl.envs import ParallelEnv
from jaxrl.env_names import get_environment_list
# import imageio.v3 as iio
from absl import app, flags
from pandas import DataFrame, read_csv
from train import FLAGS

os.environ['MUJOCO_GL'] = 'egl'

episode_len = 900
flag = flags.FLAGS
flags.DEFINE_string('ckp', 'HB_NOHANDS', 'Name of the environment to use.')
flags.DEFINE_string('robot', 'cartpole', 'Name of the robot to use.')


def main(_):
    submit_dir = os.environ.get('SLURM_SUBMIT_DIR') if os.environ.get('SLURM_SUBMIT_DIR') is not None else '.'

    out_path = submit_dir + '/summary.csv'
    if not os.path.exists(out_path):
        summary = DataFrame(columns=['task', 'num_seeds', 'goal', 'return'])
    else:
        summary = read_csv(out_path)
    # save_dir = r'/pfs/work9/workspace/scratch/ka_et4232-tcx/checkpoints'
    save_dir = './checkpoints'

    for i, task in enumerate(os.listdir(save_dir)):
        if not FLAGS.robot == task.split('-')[0]:
            continue
        checkpoint_name = task
        task_path = f'{save_dir}/{task}'
        env_name = str(checkpoint_name)
        env_names = get_environment_list(env_name)
        num_tasks = len(env_names)

        env = ParallelEnv(env_names, seed=0)

        kwargs = {}
        kwargs['updates_per_step'] = 2
        kwargs['width_critic'] = 1

        agent = BRC(
            0,
            env.observation_space.sample()[:1],
            env.action_space.sample()[:1],
            num_tasks=num_tasks,
            **kwargs,
        )
        idx =len(summary)
        summary.loc[idx] = [task, len(os.listdir(task_path)), 0., 0.]

        for seeds in os.listdir(task_path):
            if not os.path.exists(f'{task_path}/{seeds}/actor.txt'):
                continue
            agent.load_inference(f'{task_path}/{seeds}')

            env.reset()
            eval_stats = env.evaluate(agent, num_episodes=1, temperature=0.0, render=True, max_render_steps=episode_len)

            renders = eval_stats['renders']
            summary.loc[idx, 'goal'] += eval_stats['goal']
            summary.loc[idx, 'return'] += eval_stats['return']
            videos_dir = f'{submit_dir}/videos/{env_name}/{seeds}'
            os.makedirs(videos_dir, exist_ok=True)
            for j in range(renders.shape[0]):
                frames = renders[j]  # shape: (num_frames, channels, height, width)
                frames = frames.transpose(0, 2, 3, 1)  # Rearrange to (num_frames, height, width, channels)
                frames = (frames * 255).astype('uint8') if frames.dtype != 'uint8' else frames
                # frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]  # Convert RGB to BGR

                # height, width = frames[0].shape[:2]
                video_path = os.path.join(videos_dir, f'task_{j}.npy')
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
                # iio.imwrite(video_path, frames, fps=60)
                with open(video_path, 'wb') as f:
                    np.save(f, frames)

            summary.iloc[idx, 2:4] = summary.iloc[idx, 2:4] / summary.iloc[i,1]
    summary.to_csv(out_path)
if __name__ == "__main__":
    app.run(main)


