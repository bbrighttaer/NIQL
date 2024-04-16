import functools
import logging
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from marllib.marl import JointQRNN
from marllib.marl.models import QMixer, VDNMixer
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import override
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.torch_ops import convert_to_torch_tensor, convert_to_non_torch_type
from ray.rllib.utils.torch_ops import huber_loss
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

from niql.config import FINGERPRINT_SIZE

logger = logging.getLogger(__name__)

NEIGHBOURS_OBS = 'neighbours_obs'
NEIGHBOURS_NEXT_OBS = 'neighbours_next_obs'


def get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


def expand_obs_space(obs: spaces.Box, expand_size: int):
    obs = spaces.Box(
        low=obs.low[0],
        high=obs.high[0],
        shape=(sum([obs.shape[0] for _ in range(expand_size + 1)]),),
        dtype=obs.dtype,
    )
    obs._np_random = obs.np_random
    return obs


class IMIX(Policy):

    def __init__(self, obs_space, action_space, config):
        self.framework = "torch"
        config = dict(DEFAULT_CONFIG, **config)
        super().__init__(obs_space, action_space, config)
        self.n_agents = len(obs_space.original_space)
        config["model"]["n_agents"] = self.n_agents
        self.max_neighbours = config["max_neighbours"]
        self.comm = None
        self.policy_id = None
        self.use_fingerprint = config.get("use_fingerprint", False)
        self.info_sharing = config.get("info_sharing", False)
        self.n_actions = action_space.n
        self.h_size = config["model"]["lstm_cell_size"]
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.obs_size = get_size(obs_space)
        self.env_global_state_shape = (self.obs_size, self.n_agents)
        config["model"]["full_obs_space"] = obs_space
        core_arch = config["model"]["custom_model_config"]["model_arch_args"]["core_arch"]
        self.q_values = []

        # models
        obs_space = config['obs_space']
        self.model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else JointQRNN
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork if core_arch == "mlp" else JointQRNN
        ).to(self.device)

        self.exploration = self._create_exploration()
        self.dist_class = TorchCategorical

        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(self._state_inputs) > 0
        self._training_iteration_num = 0
        self._global_update_count = 0

        # Setup the mixer network.
        custom_config = config["model"]["custom_model_config"]
        custom_config["num_agents"] = self.max_neighbours + 1
        state_dim = custom_config["space_obs"]["obs"].shape
        if config["mixer"] is None:  # "iql"
            self.mixer = None
            self.target_mixer = None
        elif config["mixer"] == "qmix":
            self.mixer = QMixer(custom_config, state_dim).to(self.device)
            self.target_mixer = QMixer(custom_config, state_dim).to(self.device)
        elif config["mixer"] == "vdn":
            self.mixer = VDNMixer().to(self.device)
            self.target_mixer = VDNMixer().to(self.device)

        # optimizer
        self.params = list(self.model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())
        if config["optimizer"] == "rmsprop":
            from torch.optim import RMSprop
            self.optimiser = RMSprop(
                params=self.params,
                lr=config["lr"])

        elif config["optimizer"] == "adam":
            from torch.optim import Adam
            self.optimiser = Adam(
                params=self.params,
                lr=config["lr"], )

        else:
            raise ValueError("choose one optimizer type from rmsprop(RMSprop) or adam(Adam)")

        # Auto-update model's inference view requirements, if recurrent.
        self._update_model_view_requirements_from_init_state()
        # Combine view_requirements for Model and Policy.
        self.view_requirements.update(self.model.view_requirements)

        # initial target network sync
        self.update_target()

    @override(Policy)
    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[AgentID, Tuple[
                "Policy", SampleBatch]]] = None,
            episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
        # observation sharing
        sample_batch = sample_batch.copy()
        shared_data_obs = []
        shared_data_next_obs = []
        for n_policy, n_data in other_agent_batches.values():
            n_data = n_data.copy()
            shared_data_obs.append(n_data[SampleBatch.OBS])
            shared_data_next_obs.append(n_data[SampleBatch.NEXT_OBS])
        sample_batch[NEIGHBOURS_OBS] = np.concatenate(shared_data_obs, axis=1)
        sample_batch[NEIGHBOURS_NEXT_OBS] = np.concatenate(shared_data_next_obs, axis=1)
        return sample_batch

    @override(Policy)
    def compute_actions(
            self,
            obs_batch: Union[List[TensorStructType], TensorStructType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
            prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        # Switch to eval mode.
        if self.model:
            self.model.eval()

        with torch.no_grad():
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            obs_batch = SampleBatch({
                SampleBatch.CUR_OBS: obs_batch
            })
            obs_batch.set_get_interceptor(
                functools.partial(convert_to_torch_tensor, device=self.device)
            )
            if prev_action_batch is not None:
                obs_batch[SampleBatch.PREV_ACTIONS] = np.asarray(prev_action_batch)
            if prev_reward_batch is not None:
                obs_batch[SampleBatch.PREV_REWARDS] = np.asarray(prev_reward_batch)
            state_batches = [
                convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
            ]

            # Call the exploration before_compute_actions hook.
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)

            # predict q-vals
            initial_state = self.model.get_initial_state()
            dist_inputs, state_out = self.model(obs_batch, state_batches, seq_lens)

            # select action
            action_dist = self.dist_class(dist_inputs, self.model)
            actions, logp = self.exploration.get_exploration_action(
                action_distribution=action_dist,
                timestep=timestep,
                explore=explore,
            )

            # assign selection actions
            obs_batch[SampleBatch.ACTIONS] = actions

            # Update our global timestep by the batch size.
            self.global_timestep += len(obs_batch[SampleBatch.CUR_OBS])

            # store q values selected in this time step for callbacks
            self.q_values = dist_inputs.squeeze().detach().numpy().tolist()

            results = convert_to_non_torch_type((actions, state_out, {'q-values': [self.q_values]}))

        return results

    def compute_single_action(self, *args, **kwargs) -> \
            Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        return super().compute_single_action(*args, **kwargs)

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        # Set Model to train mode.
        if self.model:
            self.model.train()

        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self,
            train_batch=samples,
            result=learn_stats,
        )

        # compute loss
        loss_out, td_error_abs, qt_selected_mean, qt_target_mean = self.compute_q_losses(samples)

        # Optimise
        self.optimiser.zero_grad()
        loss_out.backward()
        grad_norm_clipping_ = self.config["grad_clip"]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, grad_norm_clipping_)
        self.optimiser.step()

        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": td_error_abs,
            "q_taken_mean": qt_selected_mean,
            "target_mean": qt_target_mean,
        }
        data = {LEARNER_STATS_KEY: stats}

        if self.model:
            data["model"] = self.model.metrics()
        data.update({"custom_metrics": learn_stats})
        return data

    @override(Policy)
    def get_initial_state(self):
        return [
            s.expand([self.n_agents, -1]).cpu().numpy()
            for s in self.model.get_initial_state()
        ]

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.target_model.load_state_dict(
            self._device_dict(weights["target_model"]))
        if weights["mixer"] is not None:
            self.mixer.load_state_dict(self._device_dict(weights["mixer"]))
            self.target_mixer.load_state_dict(
                self._device_dict(weights["target_mixer"]))

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
            "mixer": self._cpu_dict(self.mixer.state_dict())
            if self.mixer else None,
            "target_mixer": self._cpu_dict(self.target_mixer.state_dict())
            if self.mixer else None,
        }

    @override(Policy)
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()
        }

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        logger.debug("Updated target networks")

    def on_global_var_update(self, global_vars: Dict[str, TensorType]) -> None:
        self._global_update_count += 1
        if self._global_update_count % self.config["timesteps_per_iteration"] == 0:
            self._training_iteration_num += 1

    def compute_q_losses(self, train_batch: SampleBatch) -> TensorType:
        assert self.mixer is not None, "Mixer is required"

        # batch preprocessing ops
        obs = self.convert_batch_to_tensor({
            SampleBatch.OBS: train_batch[SampleBatch.OBS]
        })
        next_obs = self.convert_batch_to_tensor({
            SampleBatch.OBS: train_batch[SampleBatch.NEXT_OBS]
        })
        neighbour_policies = train_batch['neighbour_policies']
        train_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )

        # get hidden states for RNN case
        i = 0
        state_batches_h = []
        while "state_in_{}".format(i) in train_batch:
            state_batches_h.append(train_batch["state_in_{}".format(i)])
            i += 1
        if self._is_recurrent:
            assert state_batches_h
        # i = 0
        # state_batches_h_prime = []
        # while "state_out_{}".format(i) in train_batch:
        #     state_batches_h_prime.append(train_batch["state_out_{}".format(i)])
        #     i += 1
        # assert state_batches_h_prime
        seq_lens = train_batch.get(SampleBatch.SEQ_LENS)

        # compute q-vals
        qt, _ = self.model(obs, state_batches_h, seq_lens)
        qt_prime, _ = self.model(next_obs, state_batches_h, seq_lens)

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS].long(), num_classes=self.action_space.n)
        qt_selected = torch.sum(qt * one_hot_selection, dim=1, keepdim=True)

        # compute estimate of the best possible value starting from state at t + 1
        dones = train_batch[SampleBatch.DONES].float()
        qt_prime_best_one_hot_selection = F.one_hot(torch.argmax(qt_prime, 1), self.action_space.n)
        qt_prime_best = torch.sum(qt_prime * qt_prime_best_one_hot_selection, dim=1, keepdim=True)
        qt_prime_best_masked = (1.0 - dones.view(-1, 1)) * qt_prime_best

        all_qt_selected = [qt_selected]
        all_qt_prime_best = [qt_prime_best_masked]

        # neighbours
        with torch.no_grad():
            for policy_id, n_policy in neighbour_policies.items():
                n_qt, _ = n_policy.model(obs, state_batches_h, seq_lens)
                n_qt = torch.sum(n_qt.detach() * one_hot_selection, dim=1, keepdim=True)
                all_qt_selected.append(n_qt)

                n_qt_prime, _ = n_policy.model(next_obs, state_batches_h, seq_lens)
                n_qt_prime_best_one_hot_selection = F.one_hot(torch.argmax(n_qt_prime, 1), self.action_space.n)
                n_qt_prime_best = torch.sum(n_qt_prime * n_qt_prime_best_one_hot_selection, dim=1, keepdim=True)
                n_qt_prime_best_masked = (1.0 - dones.view(-1, 1)) * n_qt_prime_best
                all_qt_prime_best.append(n_qt_prime_best_masked)

        # prepare for mixing
        chosen_action_qvals = torch.cat(all_qt_selected, dim=1)
        target_max_qvals = torch.cat(all_qt_prime_best, dim=1)
        state = torch.cat([train_batch[SampleBatch.OBS], train_batch[NEIGHBOURS_OBS]], dim=1)
        next_state = torch.cat([train_batch[SampleBatch.NEXT_OBS], train_batch[NEIGHBOURS_NEXT_OBS]], dim=1)

        # nonlinear mixing
        chosen_action_qvals = self.mixer(chosen_action_qvals, state).squeeze()
        target_max_qvals = self.target_mixer(target_max_qvals, next_state).squeeze()

        # compute RHS of bellman equation
        qt_target = train_batch[SampleBatch.REWARDS] + self.config["gamma"] * target_max_qvals

        # Compute the error (Square/Huber).
        td_error = chosen_action_qvals - qt_target.detach()
        loss = torch.mean(huber_loss(td_error))

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        self.model.tower_stats["loss"] = loss

        # TD-error tensor in final stats
        # will be concatenated and retrieved for each individual batch item.
        self.model.tower_stats["td_error"] = td_error

        return loss, td_error.abs().sum().item(), chosen_action_qvals.mean().item(), target_max_qvals.mean().item()

    def convert_batch_to_tensor(self, data_dict):
        obs_batch = SampleBatch(data_dict)
        obs_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )
        return obs_batch

    def _pad_observation(self, obs_batch):
        """
        Add training iteration number and exploration value to observation.
        obs_batch is structured as [batch_size, feature_dim, ...]
        """
        b = obs_batch.shape[0]
        fp = np.array(
            [
                self.exploration.get_state()['cur_epsilon'],
                self._training_iteration_num
            ]
        ).reshape(1, -1)
        fp = np.tile(fp, (b, 1))
        obs_batch = np.concatenate([obs_batch[:, :-FINGERPRINT_SIZE], fp], axis=1)
        return obs_batch

    def _pad_sample_batch(self, sample_batch: SampleBatch):
        sample_batch = sample_batch.copy()
        obs = sample_batch[SampleBatch.OBS]
        obs = self._pad_observation(obs)
        next_obs = sample_batch[SampleBatch.NEXT_OBS]
        next_obs = self._pad_observation(next_obs)
        sample_batch[SampleBatch.OBS] = obs
        sample_batch[SampleBatch.NEXT_OBS] = next_obs
        return sample_batch

    def start_consensus(self, self_policy_id: str, neighbour_policies: Dict[str, Policy]):
        """
        Simulates consensus communication between agents.

        :param self_policy_id: ID of current policy
        :param neighbour_policies: Neighbour policies.
        :return:
        """
        print('start_consensus')
