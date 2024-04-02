import functools
import logging
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.torch_ops import huber_loss
from marllib.marl import JointQRNN
from ray.rllib import Policy, SampleBatch
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import override
from ray.rllib.utils.torch_ops import convert_to_torch_tensor, convert_to_non_torch_type
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

logger = logging.getLogger(__name__)


def get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


class IQLPolicy(Policy):

    def __init__(self, obs_space, action_space, config):
        self.framework = "torch"
        config = dict(DEFAULT_CONFIG, **config)
        super().__init__(obs_space, action_space, config)
        self.n_agents = len(obs_space.original_space)
        config["model"]["n_agents"] = self.n_agents
        self.n_actions = action_space.n
        self.h_size = config["model"]["lstm_cell_size"]
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.obs_size = get_size(obs_space)
        self.env_global_state_shape = (self.obs_size, self.n_agents)
        config["model"]["full_obs_space"] = obs_space
        self.core_arch = config["model"]["custom_model_config"]["model_arch_args"]["core_arch"]

        # models
        self.model = ModelCatalog.get_model_v2(
            config['obs_space'],
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork  # if core_arch == "mlp" else JointQRNN
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            config['obs_space'],
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=FullyConnectedNetwork  # if core_arch == "mlp" else JointQRNN
        ).to(self.device)

        self.exploration = self._create_exploration()
        self.dist_class = TorchCategorical

        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(self._state_inputs) > 0

        # optimizer
        self.params = self.model.parameters()
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
        self._is_recurrent = state_batches is not None and state_batches != []

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

            results = convert_to_non_torch_type((actions, state_out, {}))

        return results

    @override(Policy)
    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[AgentID, Tuple[
                "Policy", SampleBatch]]] = None,
            episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
        return sample_batch

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
        self.target_model.load_state_dict(self._device_dict(weights["target_model"]))

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
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
        logger.debug("Updated target networks")

    def compute_q_losses(self, train_batch: SampleBatch) -> TensorType:
        # batch preprocessing ops
        obs = self.convert_batch_to_tensor({
            SampleBatch.OBS: train_batch[SampleBatch.OBS]
        })
        next_obs = self.convert_batch_to_tensor({
            SampleBatch.OBS: train_batch[SampleBatch.NEXT_OBS]
        })
        train_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )

        # compute q-vals
        qt, _ = self.model(obs, [], None)
        qt_prime, _ = self.model(next_obs, [], None)

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS].long(), num_classes=self.action_space.n)
        qt_selected = torch.sum(qt * one_hot_selection, dim=1)

        # compute estimate of the best possible value starting from state at t + 1
        dones = train_batch[SampleBatch.DONES].float()
        qt_prime_best_one_hot_selection = F.one_hot(torch.argmax(qt_prime, 1), self.action_space.n)
        qt_prime_best = torch.sum(qt_prime * qt_prime_best_one_hot_selection, 1)
        qt_prime_best_masked = (1.0 - dones) * qt_prime_best

        # compute RHS of bellman equation
        qt_target = train_batch[SampleBatch.REWARDS] + self.config["gamma"] * qt_prime_best_masked

        # Compute the error (Square/Huber).
        td_error = qt_selected - qt_target.detach()
        loss = torch.mean(huber_loss(td_error))

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        self.model.tower_stats["loss"] = loss
        # TD-error tensor in final stats
        # will be concatenated and retrieved for each individual batch item.
        self.model.tower_stats["td_error"] = td_error

        return loss, td_error.abs().sum().item(), qt_selected.mean().item(), qt_target.mean().item()

    def convert_batch_to_tensor(self, data_dict):
        obs_batch = SampleBatch(data_dict)
        obs_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=self.device)
        )
        return obs_batch
