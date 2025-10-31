import os

import numpy as np
import pandas

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
flags.DEFINE_string('robot', 'h1', 'Name of the robot to use.')
flags.DEFINE_string('task', '', 'Name of whole task')

def main(_):
    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        submit_dir = os.environ.get('SLURM_SUBMIT_DIR')
    else:
        submit_dir = '.'
    save_dir = r'/pfs/work9/workspace/scratch/ka_et4232-tcx/checkpoints'
    if not os.path.exists(save_dir):
        save_dir = './checkpoints'

    out_path = submit_dir + '/summary.csv'
    if not os.path.exists(out_path):
        summary = DataFrame(columns=['task', 'num_seeds', 'goal', 'return'])
    else:
        summary = read_csv(out_path)

    if FLAGS.task != '':
        tasks =  str.split(FLAGS.task, ' ')
    else:
        ckps = os.listdir(save_dir)
        tasks = []
        for i, task in enumerate(ckps):
            if FLAGS.robot == task.split('-')[0]:
                tasks.append(task)

    entry_list = []
    to_remove_idx = []
    for index, row in summary.iterrows():
        if row['task'] in tasks:
            to_remove_idx.append(index)
    df = summary.drop(to_remove_idx)

    for i, task in enumerate(tasks):
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
        mean_goal = 0
        mean_return = 0
        num_seeds = len(os.listdir(task_path))

        for seeds in os.listdir(task_path):
            if not os.path.exists(f'{task_path}/{seeds}/actor.txt'):
                num_seeds -= 1
                continue
            agent.load_inference(f'{task_path}/{seeds}')

            env.reset()
            eval_stats = env.evaluate(agent, num_episodes=1, temperature=0.0, render=True, max_render_steps=episode_len)

            renders = eval_stats['renders']
            mean_goal += eval_stats['goal']
            mean_return += eval_stats['return']
            videos_dir = f'{submit_dir}/videos/{env_name}/{seeds}'
            os.makedirs(videos_dir, exist_ok=True)
            for j in range(renders.shape[0]):
                frames = renders[j]  # shape: (num_frames, channels, height, width)
                frames = frames.transpose(0, 2, 3, 1)  # Rearrange to (num_frames, height, width, channels)
                frames = (frames * 255).astype('uint8') if frames.dtype != 'uint8' else frames
                video_path = os.path.join(videos_dir, f'task_{j}.npy')
                with open(video_path, 'wb') as f:
                    np.save(f, frames)
        mean_goal /= num_seeds
        mean_return /= num_seeds
        entry_list.append({'task':task, 'num_seeds':num_seeds, 'goal': mean_goal, 'return':mean_return})

    df = pandas.concat([df, DataFrame(entry_list)], ignore_index=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    app.run(main)


