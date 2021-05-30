import sys
import numpy as np
import torch
import torch.nn as nn
import collections
import time


class RewardTracker:
    # adapted from < https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter10/lib/common.py>
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        # speed = (frame - self.ts_frame) / (time.time() - self.ts)
        # self.ts_frame = frame
        # self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        
        if len(self.total_rewards) % 50 == 0:
            print(
                " done %d games, mean reward %.3f "
                % (len(self.total_rewards), mean_reward)
            )
            sys.stdout.flush()

        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved !")
            return True
        return False


class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB

    Designed and tested with pytorch-tensorboard in mind
    """

    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(
            value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)
        ) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()


def unpack_batch(batch, net, params):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    device = params["device"]

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))

        expanded_action = []
        for act in exp.action:
            if act > 0:
                expanded_action.extend([0, 1])
            else:
                expanded_action.extend([1, 0])

        actions.append((expanded_action))
        rewards.append(exp.reward)

        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= params["gamma"] ** 2
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_t, ref_vals_v


Experience = collections.namedtuple(
    "Experience", ["state", "action", "reward", "done", "time_step"]
)


class ExperienceSource:
    # adapted from https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries

    Attributes
    ----------
    steps_count: int
        the length of subtrajectories to be generated.


    """

    def __init__(self, env, agent, params):  # steps_count=4, steps_delta=2):

        assert params["reward_steps"] >= 1

        self.agent = agent
        self.steps_count = params["reward_steps"]
        self.steps_delta = params["steps_delta"]
        self.total_rewards = []
        self.total_steps = []
        self.n_tanks = env.n_tanks
        self.env = env

    def __iter__(self):
        states, agent_states, total_rewards, total_steps = [], [], [], []
        history = collections.deque(maxlen=self.steps_count)
        obs = self.env.reset()
        cur_rewards = 0
        cur_steps = 0
        agent_states = self.agent.initial_state()
        iter_idx = 0
        while True:

            actions, agent_states = self.agent(obs, agent_states)

            obs, reward, is_done, info = self.env.step(actions, cur_steps)

            r = sum(reward)
            cur_rewards += r
            cur_steps += 1
            history.append(
                Experience(
                    state=obs,
                    action=actions,
                    reward=r,
                    done=is_done,
                    time_step=cur_steps,
                )
            )

            if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                yield tuple(history)
            if is_done:
                if 0 < len(history) < self.steps_count:
                    yield tuple(history)
                while len(history) > 1:
                    history.popleft()
                    yield tuple(history)

                self.total_rewards.append(cur_rewards)
                self.total_steps.append(cur_steps)
                cur_rewards = 0
                cur_steps = 0
                obs = self.env.reset()
                history.clear()
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


ExperienceFirstLast = collections.namedtuple(
    "ExperienceFirstLast", ("state", "action", "reward", "last_state","epi_step")
)


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """

    def __init__(
        self, env, agent, params
    ):  # gamma=0.99, steps_count=4, steps_delta=1):

        super(ExperienceSourceFirstLast, self).__init__(env, agent, params)
        self.gamma = params["gamma"]
        self.steps = params["reward_steps"]

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
   
            yield ExperienceFirstLast(
                state=exp[0].state,
                action=exp[0].action,
                reward=total_reward,
                last_state=last_state,
                epi_step=exp[-1].time_step,
            )
