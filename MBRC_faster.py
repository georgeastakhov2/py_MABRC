import numpy as np
import pandas as pd
from shock_generator import generate_shock

print(__name__)
def MABRC_Simulation(
        vola: np.ndarray,
        riskfree: np.ndarray,
        div: np.ndarray,
        spot: np.ndarray,
        barrier: float,
        T: float,
        iterations: int,
        cormat,
        am_type: bool,
        margin=1.0,
):
    nom = 1000
    n = np.shape(spot)[0]
    b_d = 260  # business days
    T = int(round(T * b_d, 0))
    t = 1 / b_d
    vola = vola * margin
    # Gathering all needed parameters for simulation within dictionaries
    sim_list = ["sim_1", "sim_2", "sim_3", "sim_4", "sim_5"]

    spots = dict((el, 0) for el in sim_list[0:n])
    spots.update(zip(spots, spot))
    barrier_spot = np.array(spot * barrier)
    barrier_spots = dict((el, 0) for el in sim_list[0:n])
    barrier_spots.update(zip(barrier_spots, barrier_spot))

    ratio = np.array(nom / spot)
    ratios = dict((el, 0) for el in sim_list[0:n])
    ratios.update(zip(ratios, ratio))

    div = (np.exp(div) - 1) * t
    divs = dict((el, 0) for el in sim_list[0:n])
    divs.update(zip(divs, div))

    chol = np.linalg.cholesky(cormat).T

    drift = (riskfree - 0.5 * vola ** 2) * t
    drifts = dict((el, 0) for el in sim_list[0:n])
    drifts.update(zip(drifts, drift))
    # Simulating Stock Prices with GBM (Geometric Brownian Motion)
    sims = {}

    for i in range(n):
        sims[sim_list[i]] = pd.DataFrame(index=range(iterations), columns=range(T))
    sim_proc = iter(sims)
    for stock in sim_proc:
        sims[stock].iloc[:, 0] = spots[stock]

    for col in range(1, T):
        sim_proc = iter(sims)
        # Shocks are simulated separately for convenience in form of matrix:
        shocks = generate_shock(iterations, n=n, vola=vola, chol=chol, t=t)
        for stock in sim_proc:
            shock = []
            for value in shocks[stock]:
                shock.append(value)
            shock = np.array(shock)
            P_t = np.array(sims[stock].iloc[:, col - 1])
            change = np.exp(drifts[stock] + shock)
            sims[stock].iloc[:, col] = P_t * (1 - divs[stock]) * change
    # Calculating payoff
    payoff = np.zeros(iterations)
    for row in range(iterations):
        sim_proc = iter(sims)
        returns = []
        touch = False
        for stock in sim_proc:
            returns.append(sims[stock].iloc[row, T - 1] / spots[stock])
            if am_type:
                current_row_min = np.array(sims[stock].iloc[row, :]).min()
                check = current_row_min <= barrier_spots[stock]
            else:
                check = returns[-1] <= barrier
            if check:
                touch = True
        returns = np.array(returns)
        if touch:
            if returns.min() < 1:
                pos = np.argmin(returns)
                payoff[row] = sims[sim_list[pos]].iloc[row, T - 1] * ratios[sim_list[pos]]
            else:
                payoff[row] = nom
        else:
            payoff[row] = nom
    # Defining fair coupon
    mean_payoff = np.mean(payoff)
    T = T / b_d
    discounted_payoff = mean_payoff * np.exp(-riskfree[0] * T)
    coupon = (1 - discounted_payoff / nom) / T
    coupon = coupon * 100
    return np.round(coupon, 2)

# vola, div, riskfree, spot, cormat, iterations, T, barrier = user_input()
# sim = MABRC_Simulation(vola=vola,
#                       riskfree=riskfree,
#                       spot=spot,
#                       iterations=iterations,
#                       T=T,
#                       barrier=barrier,
#                       div=div,
#                       cormat=cormat,
#                       am_type=True
#                       )
#
#
# print(sim)