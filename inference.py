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
flags.DEFINE_string('robot', 'cheetah', 'Name of the robot to use.')
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
        tasks =  enumerate(str.split(FLAGS.task, ' '))
    else:
        tasks = enumerate(os.listdir(save_dir))
    for i, task in tasks:
        if not FLAGS.robot == task.split('-')[0]:
            continue

        existing_record = summary[summary['task'] == task]
        checkpoint_name = task
        task_path = f'{save_dir}/{task}'
        env_name = str(checkpoint_name)
        env_names = get_environment_list(env_name)
        num_tasks = len(env_names)

        if FLAGS.task == '' and len(existing_record) == 1 and existing_record.iloc[0,1] == len(os.listdir(task_path)):
            continue



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

        summary = summary[summary['task'] != task]
        idx =len(summary)
        summary.loc[idx] = [task, len(os.listdir(task_path)), 0., 0.]

        for seeds in os.listdir(task_path):
            if not os.path.exists(f'{task_path}/{seeds}/actor.txt'):
                summary.iloc[-1, 1] -= 1
                continue
            agent.load_inference(f'{task_path}/{seeds}')

            env.reset()
            eval_stats = env.evaluate(agent, num_episodes=1, temperature=0.0, render=True, max_render_steps=episode_len)

            renders = eval_stats['renders']
            summary.iloc[-1, 2] += eval_stats['goal']
            summary.iloc[-1, 3] += eval_stats['return']
            videos_dir = f'{submit_dir}/videos/{env_name}/{seeds}'
            os.makedirs(videos_dir, exist_ok=True)
            for j in range(renders.shape[0]):
                frames = renders[j]  # shape: (num_frames, channels, height, width)
                frames = frames.transpose(0, 2, 3, 1)  # Rearrange to (num_frames, height, width, channels)
                frames = (frames * 255).astype('uint8') if frames.dtype != 'uint8' else frames
                video_path = os.path.join(videos_dir, f'task_{j}.npy')
                with open(video_path, 'wb') as f:
                    np.save(f, frames)
        summary.iloc[-1, 2:4] = summary.iloc[-1, 2:4] / summary.iloc[-1,1]
    summary.to_csv(out_path, index=False)


if __name__ == "__main__":
    app.run(main)


