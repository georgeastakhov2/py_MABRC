import pandas as pd
import numpy as np


def generate_shock(iterations, n, vola, chol, t):
    sim_list = ["sim_1", "sim_2", "sim_3", "sim_4", "sim_5"]
    random = np.array(np.random.standard_normal(size=(iterations, n)))
    corr_shock = np.dot(random, chol)
    shock = vola * corr_shock * np.sqrt(t)
    shock = pd.DataFrame(shock, columns=sim_list[0:n])
    shock_dict = shock.to_dict(orient="list")
    return shock_dict
