import logging

import ray
import torch

from niql import envs, config
from niql.eval import load_model

logger = logging.getLogger(__name__)

# get env
env_instance, env_info = envs.make_matrix_game_env()
config = config.COOP_MATRIX

# load saved checkpoint
base = 'exp_results/iql_mlp_all_scenario/IQL_CoopMatrixGame_all_scenario_5e48e_00000_0_2024-04-09_03-28-09'
ckpt = load_model({
    f'params_path': f'{base}/params.json',  # experiment configuration
    'model_path': f'{base}/checkpoint_000010/checkpoint-10',  # checkpoint path
})
agent, pmap = ckpt.trainer, ckpt.pmap

# Inference
for _ in range(1):
    obs = env_instance.reset()
    done = {"__all__": False}
    states = {
        actor_id: agent.get_policy(pmap(actor_id)).get_initial_state()
        for actor_id in obs
    }

    step = 0
    with torch.no_grad():
        while not done["__all__"]:
            action_dict = {}
            for agent_id in obs.keys():
                policy = agent.get_policy(pmap(agent_id))
                agent_obs = [[0, 0, 0]] if step == 0 else [[0, 0, 1]]  #  obs[agent_id]["obs"]
                action_dict[agent_id], states[agent_id], info = policy.compute_single_action(
                    agent_obs,
                    states[agent_id],
                    explore=False,
                )
                print(f'state={step}, agent={agent_id}, q-values={info["q-values"]}')

            obs, reward, done, info = env_instance.step(action_dict)
            step += 1

env_instance.close()
ray.shutdown()
logger.info("Inference finished!")
