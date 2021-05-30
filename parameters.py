# -----------------RL parameters-----------
RL_params = {
    "max_time": 200,  # max simulation time
    "hidden_size_nn": 8,
    "entropy_beta": 0.01,
    "gamma": 0.99,
    "learning_rate": 0.001,
    "reward_steps": 4,
    "steps_delta": 1,
    "batch_size": 128,
    "clip_grad": 0.1,
    "device": "cpu",
}

# -----------------Tank_params------------
Tank_params = {
    "g": 9.81,
    "rho": 1000,
    "height": 10,
    "radius": 5,
    "max_lvl": 0.75,
    "min_lvl": 0.25,
    "int_lvl": 0.5,
    "disturbance": True,
    "connected_tank": False,
    "pipe_radius": 0.25,
}

# -----------------flowDisturbance
Disturbance_params = {
    "pre_def_dist": False,
    "mean_flow": 1,
    "var_flow": 0.1,
    "max_flow": 2,
    "min_flow": 0.7,
    "add_step": False,
    "max_time": int(RL_params["max_time"] / 2),
}
