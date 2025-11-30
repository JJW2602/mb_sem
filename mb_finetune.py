#Compare MBRL vs APT + MBRL

import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path


import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler

torch.backends.cudnn.benchmark = True


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.domain, _ = self.cfg.task.split('_', 1)

        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.task, cfg.agent.name, cfg.obs_type, str(cfg.snapshot_ts),
                str(cfg.seed), 
            ])
            wandb.init(project=cfg.wandb_project, group=cfg.agent.name, name=exp_name)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # Ensemble models(MB)
        state_size = np.prod(self.train_env.observation_space.shape)
        action_size = np.prod(self.train_env.action_space.shape)
        self.env_model = EnsembleDynamicsModel(cfg.num_networks, cfg.num_elites, state_size, action_size, cfg.reward_size, cfg.pred_hidden_size,
                                          use_decay=cfg.use_decay)
    
        # Predict env(MB)
        self.predict_env = PredictEnv(self.env_model, cfg.task)

        # Replay Buffer(MB)
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # Sampler(MB)
        self.env_sampler = EnvSampler(self.train_env, max_path_length=cfg.max_path_length)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.rollouts_per_epoch = cfg.rollout_batch_size * cfg.epoch_length / cfg.model_train_freq
        self.model_steps_per_epoch = int(1 * self.rollouts_per_epoch)
        self.new_pool_size = cfg.model_retain_epochs * self.model_steps_per_epoch

        # create data storage
        self.env_replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')
        self.model_replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')
        # create replay buffer
        self.env_replay_loader = make_replay_loader(self.env_replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        
        self.model_replay_loader = make_replay_loader(self.model_replay_storage,
                                                self.new_pool_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None
        self._env_replay_iter = None
        self._model_replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.domain else 2,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.total_train_steps = self.cfg.num_train_frames // self.cfg.action_repeat
        self.pbar = tqdm(
            total=self.total_train_steps,
            desc="train",
            unit="step",
            dynamic_ncols=True,
            ascii=True,
            mininterval=0.5,
            smoothing=0.1,
            disable=bool(int(os.environ.get('DISABLE_TQDM', '0'))) 
        )
        self.eval_calls = 0
        self.total_eval_episodes_run = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def env_replay_iter(self):
        if self._env_replay_iter is None:
            self._env_replay_iter = iter(self.env_replay_loader)
        return self._env_replay_iter
    
    @property
    def model_replay_iter(self):
        if self._model_replay_iter is None:
            self._model_replay_iter = iter(self.replay_loader)
        return self._model_replay_iter
    
    def video_eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

    def train_predict_model(args, env_pool, predict_env):
        # Get all samples from environment
        state, action, reward, next_state, done = env_pool.sample(len(env_pool))
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
        predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

    def set_rollout_length(cfg, epoch_step):
        rollout_length = (min(max(cfg.rollout_min_length + (epoch_step - cfg.rollout_min_epoch)
                                / (cfg.rollout_max_epoch - cfg.rollout_min_epoch) * (cfg.rollout_max_length - cfg.rollout_min_length),
                                cfg.rollout_min_length), cfg.rollout_max_length))
        return int(rollout_length)
    
    def resize_model_pool(self, cfg, rollout_length, model_pool):
        rollouts_per_epoch = cfg.rollout_batch_size * cfg.epoch_length / cfg.model_train_freq
        model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
        new_pool_size = cfg.model_retain_epochs * model_steps_per_epoch

        sample_all = model_pool.return_all()
        new_model_pool = make_replay_loader(self.model_replay_storage,
                                                new_pool_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        
        new_model_pool.push_batch(sample_all)

        return new_model_pool

    def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
        state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
        for i in range(rollout_length):
            # TODO: Get a batch of actions
            action = agent.select_action(state)
            next_states, rewards, terminals, info = predict_env.step(state, action)
            # TODO: Push a batch of samples
            model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
            nonterm_mask = ~terminals.squeeze(-1)
            if nonterm_mask.sum() == 0:
                break
            state = next_states[nonterm_mask]

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                total_reward += time_step.reward
                step += 1

            episode += 1

        avg_return = total_reward / episode
        avg_length = step * self.cfg.action_repeat / episode

        if self.cfg.use_wandb:
            wandb.log({
                'eval/episode_reward': float(avg_return),
                'eval/episode_length': float(avg_length),
                'eval/episode': int(self.global_episode),
                'eval/step': int(self.global_step),
                'global_frame': int(self.global_frame)
            }, step=self.global_frame)

        self.eval_calls += 1
        self.total_eval_episodes_run += episode

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        video_eval_every_step = utils.Every(self.cfg.video_eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.env_replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.env_replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            if video_eval_every_step(self.global_step):
                self.video_eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            if hasattr(self.agent, "regress_meta"):
                repeat = self.cfg.action_repeat
                every = self.agent.update_task_every_step // repeat
                init_step = self.agent.num_init_steps
                if self.global_step > (
                        init_step // repeat) and self.global_step % every == 0:
                    meta = self.agent.regress_meta(self.replay_iter,
                                                   self.global_step)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)
            
            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.env_replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

            # try to update model(Every epoch)
            if not seed_until_step(self.global_step):
                if self.global_step % self.cfg.model_train_freq== 0:
                    self.train_predict_model(self.cfg, self.env_replay_storage, self.predict_env)
                     # k-rollout
                    epoch_step = self.global_step // self.cfg.num_steps_per_epoch
                    new_rollout_length = self.set_rollout_length(self.cfg, epoch_step)
                    if rollout_length != new_rollout_length:
                        rollout_length = new_rollout_length
                        model_pool = self.resize_model_pool(self.cfg, rollout_length, model_pool)

                    self.rollout_model(self.cfg, self.predict_env, self.agent, self.model_replay_storage, self.env_replay_storage, rollout_length)

            # try to update the agent
            if not seed_until_step(self.global_step):
                for _ in range(self.cfg.policy_update_per_step):
                    metrics = self.agent.update(self.model_replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')


            self.pbar.update(1)
            self.pbar.set_postfix(ep=self._global_episode,
                                  ep_reward=float(episode_reward))
        self.pbar.close()

        if self.cfg.use_wandb:
            wandb.summary["total_train_steps"] = int(self.global_step)
            wandb.summary["total_train_episodes"] = int(self.global_episode)
            wandb.summary["total_eval_calls"] = int(self.eval_calls)
            wandb.summary["total_eval_episodes"] = int(self.total_eval_episodes_run)

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        snapshot_dir = snapshot_base_dir / self.domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = snapshot_dir / f'seed_{seed}' / 'snapshot' / f'snapshot_{self.cfg.snapshot_ts}.pt'
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None


@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    from finetune import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
