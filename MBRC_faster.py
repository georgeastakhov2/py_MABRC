import numpy as np
import pandas as pd
from shock_generator import generate_shock

def simulate_mabrc(
    volatility: np.ndarray,
    risk_free_rate: np.ndarray,
    dividend_yield: np.ndarray,
    initial_spot_prices: np.ndarray,
    barrier_level: float,
    time_to_maturity: float,
    num_iterations: int,
    correlation_matrix: np.ndarray,
    is_american_type: bool,
    volatility_margin: float = 1.0,
) -> float:
    """
    Simulates a Multi-Asset Barrier Reverse Convertible (MABRC) and calculates the fair coupon.

    Args:
        volatility: Array of volatilities for each asset.
        risk_free_rate: Array of risk-free rates.
        dividend_yield: Array of dividend yields for each asset.
        initial_spot_prices: Array of initial spot prices for each asset.
        barrier_level: Barrier level as a fraction of the initial spot price.
        time_to_maturity: Time to maturity in years.
        num_iterations: Number of Monte Carlo simulations.
        correlation_matrix: Correlation matrix between assets.
        is_american_type: True for American-style barrier, False for European-style.
        volatility_margin: Margin applied to volatilities. Defaults to 1.0.

    Returns:
        The fair coupon rate as a percentage.
    """

    nominal_value = 1000
    num_assets = len(initial_spot_prices)
    business_days_per_year = 260
    time_steps = int(round(time_to_maturity * business_days_per_year))
    time_step_size = 1 / business_days_per_year
    adjusted_volatility = volatility * volatility_margin

    asset_names = [f"asset_{i+1}" for i in range(num_assets)]

    asset_data = {
        "initial_spot": dict(zip(asset_names, initial_spot_prices)),
        "barrier_spot": dict(zip(asset_names, initial_spot_prices * barrier_level)),
        "ratio": dict(zip(asset_names, nominal_value / initial_spot_prices)),
        "dividend": dict(zip(asset_names, (np.exp(dividend_yield) - 1) * time_step_size)),
        "drift": dict(zip(asset_names, (risk_free_rate - 0.5 * adjusted_volatility ** 2) * time_step_size)),
    }

    cholesky_decomposition = np.linalg.cholesky(correlation_matrix).T

    simulated_prices = {
        asset_name: pd.DataFrame(index=range(num_iterations), columns=range(time_steps))
        for asset_name in asset_names
    }

    for asset_name in asset_names:
        simulated_prices[asset_name].iloc[:, 0] = asset_data["initial_spot"][asset_name]

    for time_step in range(1, time_steps):
        shocks = generate_shock(num_iterations, num_assets, adjusted_volatility, cholesky_decomposition, time_step_size)
        for asset_index, asset_name in enumerate(asset_names):
            previous_prices = simulated_prices[asset_name].iloc[:, time_step - 1].values
            price_changes = np.exp(asset_data["drift"][asset_name] + shocks[asset_index])
            simulated_prices[asset_name].iloc[:, time_step] = previous_prices * (1 - asset_data["dividend"][asset_name]) * price_changes

    payoffs = np.zeros(num_iterations)

    for iteration in range(num_iterations):
        final_returns = [
            simulated_prices[asset_name].iloc[iteration, time_steps - 1] / asset_data["initial_spot"][asset_name]
            for asset_name in asset_names
        ]

        barrier_touched = False
        for asset_name in asset_names:
            if is_american_type:
                min_price = simulated_prices[asset_name].iloc[iteration, :].min()
                barrier_touched = barrier_touched or (min_price <= asset_data["barrier_spot"][asset_name])
            else:
                barrier_touched = barrier_touched or (final_returns[asset_names.index(asset_name)] <= barrier_level)

        if barrier_touched:
            if min(final_returns) < 1:
                worst_asset_index = np.argmin(final_returns)
                payoffs[iteration] = simulated_prices[asset_names[worst_asset_index]].iloc[iteration, time_steps - 1] * asset_data["ratio"][asset_names[worst_asset_index]]
            else:
                payoffs[iteration] = nominal_value
        else:
            payoffs[iteration] = nominal_value

    average_payoff = np.mean(payoffs)
    discounted_payoff = average_payoff * np.exp(-risk_free_rate[0] * time_to_maturity)
    coupon_rate = (1 - discounted_payoff / nominal_value) / time_to_maturity * 100

    return round(coupon_rate, 2)