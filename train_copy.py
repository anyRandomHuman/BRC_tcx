import os

os.environ['MUJOCO_GL'] = 'egl'

from absl import app, flags

from jaxrl.agent.brc_learner import BRC
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.envs import ParallelEnv
from jaxrl.normalizer import RewardNormalizer
from jaxrl.logger import EpisodeRecorder
from jaxrl.env_names import get_environment_list

# from OpenGL import EGL
# EGL.eglInitialize(EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY), None, None)


FLAGS = flags.FLAGS


# flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_integer('eval_episodes', 1, 'Number of episodes used for evaluation.')
# flags.DEFINE_integer('eval_interval', 30, 'Eval interval.')
# flags.DEFINE_integer('batch_size', 20, 'Mini batch size.')
# flags.DEFINE_integer('max_steps', 100, 'Number of training steps.')
# flags.DEFINE_integer('replay_buffer_size', 10, 'Replay buffer size.')
# flags.DEFINE_integer('start_training', 15,'Number of training steps to start training.')
# flags.DEFINE_string('env_names', 'cartpole-swingup', 'Environment name.')
# flags.DEFINE_boolean('log_to_wandb', True, 'Whether to log to wandb.')
# flags.DEFINE_boolean('offline_evaluation', False, 'Whether to perform evaluations with temperature=0.')
# flags.DEFINE_boolean('render', False, 'Whether to log the rendering to wandb.')
# flags.DEFINE_integer('updates_per_step', 1, 'Number of updates per step.')
# flags.DEFINE_integer('width_critic', 4, 'Width of the critic network.')
# flags.DEFINE_string('save_location', './checkpoints', 'path to save checkpoints, need to be absolute if on cluster')
# flags.DEFINE_integer('assigned_time', 64800, 'Width of the critic network.')

flags.DEFINE_string('test', 'False', 'Whether to run in test mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1000000), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(1000000), 'Replay buffer size.')
flags.DEFINE_integer('start_training', int(5000),'Number of training steps to start training.')
flags.DEFINE_string('env_names', 'dog-run', 'Environment name.')
flags.DEFINE_boolean('log_to_wandb', True, 'Whether to log to wandb.')
flags.DEFINE_boolean('offline_evaluation', True, 'Whether to perform evaluations with temperature=0.')
flags.DEFINE_boolean('render', False, 'Whether to log the rendering to wandb.')
flags.DEFINE_integer('updates_per_step', 2, 'Number of updates per step.')
flags.DEFINE_integer('width_critic', 4096, 'Width of the critic network.')
flags.DEFINE_integer('assigned_time', 64800, 'Width of the critic network.')

def main(_):
    print(f'task: {FLAGS.env_names}')
    if FLAGS.log_to_wandb:
        import wandb
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        wandb.init(
            config=FLAGS,
            entity='',
            project=f'BRC_test',
            group=f'{FLAGS.env_names}',
            name=f'{FLAGS.env_names}_{current_time}_{FLAGS.seed}'
        )

    env_names = get_environment_list(FLAGS.env_names)
    env = ParallelEnv(env_names, seed=FLAGS.seed)
    if FLAGS.offline_evaluation:
        eval_env = ParallelEnv(env_names, seed=FLAGS.seed + 42)
    else:
        eval_env = None

    eval_interval = FLAGS.eval_interval if FLAGS.offline_evaluation else 5000
        
    # Kwargs setup
    kwargs = {}
    kwargs['updates_per_step'] = FLAGS.updates_per_step
    kwargs['width_critic'] = FLAGS.width_critic
    
    num_tasks = len(env.envs)

    agent = BRC(
        FLAGS.seed,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        **kwargs,
    )
    
    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_tasks=num_tasks)    
    reward_normalizer = RewardNormalizer(num_tasks, agent.target_entropy, discount=agent.discount, max_steps=None) #change max_steps according to env
    statistics_recorder = EpisodeRecorder(num_tasks)

    def sample(i, observations=None):          
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions(observations, temperature=1.0) #sample for every env
        next_observations, rewards, terms, truns, goals = env.step(actions) #state shape (num_envs, obs_dim)
        reward_normalizer.update(rewards, terms, truns)
        statistics_recorder.update(rewards, goals, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_observations)
        observations = next_observations
        observations, terms, truns = env.reset_where_done(observations, terms, truns)
        return observations

    if os.environ.get('SLURM_SUBMIT_DIR'):
        submit_dir = os.environ.get('SLURM_SUBMIT_DIR')
        save_space = r'/pfs/work9/workspace/scratch/ka_et4232-tcx'
    else:
        submit_dir = '.'
        save_space = '.'
    save_dir = save_space + '/checkpoints'
    #   save_dir = FLAGS.save_location
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f'{save_dir}/{FLAGS.env_names}_test'
    os.makedirs(save_path, exist_ok=True)

    obs = env.reset()
    start_iter = 0
    if os.path.exists(f'{save_path}/pause.txt'):
        try:
            agent.load(save_path)
            replay_buffer.load(save_path)
            print(f'Loaded from {save_path}, resuming from iteration {start_iter}')
            obs = sample(start_iter, obs)
        except:
            for i in range(FLAGS.start_training):
                obs = sample(i, obs)
    else:
        for i in range(FLAGS.start_training):
            obs = sample(i, obs)

    import  time
    start_time = time.time()
    pause_iter = -1
    with open(f'{save_path}/pause.txt', 'w') as f:
        for i in range(FLAGS.max_steps - FLAGS.start_training - start_iter):
            run_time = time.time() - start_time
            if os.path.exists(f'{submit_dir}/pause_test.flag'):
                pause_iter = i
                os.remove(f'{submit_dir}/pause_test.flag')
                break
            if FLAGS.assigned_time - run_time < 180:
                replay_buffer.save(save_path)
                break
            obs = sample(i + FLAGS.start_training, obs)
            batches = replay_buffer.sample(FLAGS.batch_size,
                                           FLAGS.updates_per_step)  # sample randomly from all data,not one per task
            batches = reward_normalizer.normalize(batches, agent.get_temperature())
            _ = agent.update(batches, FLAGS.updates_per_step, i)
            if i % eval_interval == 0 and i >= FLAGS.start_training:
                info_dict = statistics_recorder.log(FLAGS, agent, replay_buffer, reward_normalizer, i, eval_env,
                                                    render=FLAGS.render)
                # print(f'info_dict: {info_dict}')
                # agent.save(save_path)
                # replay_buffer.save(save_path)
                f.write(f'{i}')
                # f.write(str(info_dict))
        f.write(f'{FLAGS.max_steps}')
    agent.save(save_path)
    # replay_buffer.save(save_path)

    # if pause_iter >= 0:
    #     with open(f'{save_path}/pause.txt', 'w') as f:
    #         f.write(f'{pause_iter}')


if __name__ == '__main__':
    app.run(main)
