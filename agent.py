import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CNet(nn.Module):
    """
    Agent MLP Net for policy and value function
    Policy net output the actions for each tank. One tank has observation size of four
    and two possible actions
    """

    def __init__(self, input_shape, hidden_size):
        super(A2CNet, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, int(input_shape / 2)),
        )

        self.value = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):

        values = self.value(state)
        policy = self.policy(state)

        return (policy), values

class Agent:
    """
    Agent selects an action according to probability of each action from the net

    """

    def __init__(
        self,
        model,
        device,
        apply_softmax=True
    ):
        self.model = model
        self.device = device
        self.apply_softmax = apply_softmax

    def initial_state(self):

        return None

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """

        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        probs_v, values_v = self.model(states)
        probs_v = F.softmax(probs_v.view(-1, 2), 1)

        probs_v_np = probs_v.data.cpu().numpy()

        actions = []
        for x in probs_v_np:
            actions.extend(np.random.choice(2, 1, p=x))

        agent_states = values_v.data.squeeze().cpu().numpy().tolist()

        return np.array(actions), agent_states
