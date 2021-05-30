import numpy as np
import math


class Tank:
    """
    Tank Simulation

    """

    def __init__(self, tank_params, disturb_params):

        self.h = tank_params["height"]
        self.r = tank_params["radius"]
        self.A = tank_params["radius"] ** 2 * np.pi

        self.init_height = tank_params["int_lvl"] * self.h  # change to random TODO
        self.max_height = tank_params["max_lvl"] * self.h
        self.min_height = tank_params["min_lvl"] * self.h
        self.diff_lvl = 0
        self.q_out = 0
        self.lvl = self.init_height

        self.pipe_area = tank_params["pipe_radius"] ** 2 * np.pi

        self.inflow_disturb = tank_params["disturbance"]
        self.tank_params = tank_params
        self.disturb_params = disturb_params
        self.disturbed_flow = [self.disturb_params["mean_flow"]]

    def get_dlvl(self, valve_action, t, prev_flowrate):
        """
        calculate the water level difference dl/dt according to
        water discharge equation.

        Returns
        -------
        tuple
            difference in water level, new water level and discharge rate
        """

        if self.inflow_disturb:
            q_in = prev_flowrate + self.get_disturbance(t)
        else:
            q_in = prev_flowrate

        self.q_out = (
            valve_action
            * self.pipe_area
            * np.sqrt(2 * self.tank_params["g"] * self.lvl)
        )

        self.diff_lvl = (q_in - self.q_out) / (np.pi * self.r ** 2)

        self.lvl += self.diff_lvl * self.h

        return self.diff_lvl, self.lvl, self.q_out

    def is_halffill(self):
        """
        return true if water is at least half fill.
        """

        fill = True if (self.lvl / self.h) >= 0.5 else False
        return fill

    def reset(self):
        """
        reset the tank water level to initial_height
        """

        self.lvl = self.init_height
        self.diff_lvl = 0
        self.disturbed_flow = [self.disturb_params["mean_flow"]]

    def get_disturbance(self, t):
        """
        calculate the disturbance of the inflow water rate
        """
        if t > self.disturb_params["max_time"]:
            self.disturbed_flow.append(self.disturb_params["max_flow"])

        disturbance = np.clip(
            np.random.normal(self.disturbed_flow[-1], self.disturb_params["var_flow"]),
            self.disturb_params["min_flow"],
            self.disturb_params["max_flow"],
        )

        self.disturbed_flow.append(disturbance)

        return self.disturbed_flow[-1]

    def __str__(self):

        return "the tank is at {self.lvl} water level with flow rate of {self.q_out}".format(
            self=self
        )


class TankEnv:
    """
    Tank Environment
    ...

    Attributes
    ----------
    tanks : list
        a list containing connected tanks
    max_height: int
        max  allowable ratio of water level to height of the tankk
    min_height: int
        min allowable ratio of water level to height of the tank
    """

    def __init__(self, tanks, max_height=0.6, min_height=0.4):
        """"""
        self.tanks = tanks
        self.n_tanks = len(tanks)

        self.done = [False] * self.n_tanks
        self.reward = [0] * self.n_tanks
        self.q_inn = [0] * (self.n_tanks + 1)

        assert max_height > min_height
        self.max_height_ratio = max_height
        self.min_height_ratio = min_height

    def step(self, valve_action, t):
        """interact with the environment

        Parameters
        ----------
        file_loc : valve_action
            if the valve is open, the value is 1 else 0
        t : int
            time-step of the simulation

        Returns
        -------
        tupe
            a tuple of observation, reward, terminated, info
        """

        obs = []
        prev_q_out = 0

        for i in range(self.n_tanks):

            lvl_diff, new_lvl, prev_q_out = self.tanks[i].get_dlvl(
                valve_action[i], t, prev_q_out
            )

            self.q_inn[i + 1] = prev_q_out

            if i == 0:
                prev_valve_action = 0
            else:
                prev_valve_action = valve_action[i - 1]

            half_fill = 0

            if self.tanks[i].is_halffill():
                half_fill = 1

            obs.extend(
                [
                    self.tanks[i].lvl / self.tanks[i].h,
                    lvl_diff,
                    half_fill,
                    prev_valve_action,
                ]
            )

            if (self.tanks[i].lvl > self.max_height_ratio * self.tanks[i].h) or (
                self.tanks[i].lvl < self.min_height_ratio * self.tanks[i].h
            ):
                self.done[i] = True
            else:
                self.done[i] = False

            self.reward[i] = 0

            if self.done[i]:
                self.reward[i] = -1/self.n_tanks
            elif (
                self.tanks[i].lvl / self.tanks[i].h > self.min_height_ratio
                and self.tanks[i].lvl / self.tanks[i].h < self.max_height_ratio
            ):
                self.reward[i] = 1
        # if sum(self.done) > math.ceil(self.n_tanks / 3):
        #     is_done = True
        # else:
        #     is_done = False
        if sum(self.done) > 0:
            is_done = True
        else:
            is_done = False
        return obs, self.reward, is_done, None

    def reset(self):
        init_obs = []
        self.done = [False] * self.n_tanks
        for i in range(self.n_tanks):
            self.tanks[i].reset()
            init_obs.extend(
                np.array([(self.tanks[i].lvl) / self.tanks[i].h, 0, 1, 0])
            )
        return init_obs  # reward, done, info can't be included

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
