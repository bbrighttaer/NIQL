from typing import List, Any

from ray import tune
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import PolicyID


def determine_multiagent_policy_mapping(exp_info, env_info):
    map_name = exp_info["env_args"]["map_name"]
    agent_name_ls = env_info["agent_name_ls"]
    policy_mapping_info = env_info["policy_mapping_info"][map_name]
    shared_policy_name = "default_policy" if exp_info["agent_level_batch_update"] else "shared_policy"
    if exp_info["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError("in {}, policy can not be shared".format(map_name))

        policies = {shared_policy_name}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: shared_policy_name)

    elif exp_info["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))
            policies = {"shared_policy"}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: "shared_policy")
        else:
            policies = {
                "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif exp_info["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
            range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(exp_info["share_policy"]))

    return policies, policy_mapping_fn


class TrainingIterationNumberUpdater:

    def __init__(self,
                 workers: WorkerSet,
                 by_steps_trained: bool = False,
                 policies: List[PolicyID] = frozenset([])):
        self.workers = workers
        self.local_worker = workers.local_worker()
        self.policies = policies

    def __call__(self, _: Any) -> None:
        def update(policy, p_id):
            if hasattr(policy, 'update_training_iter_number'):
                policy.update_training_iter_number()
        self.workers.local_worker().foreach_trainable_policy(update)
