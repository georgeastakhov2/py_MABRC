import pandas as pd
import numpy as np
from typing import Dict, List

def generate_shock(
    num_iterations: int,
    num_assets: int,
    volatilities: np.ndarray,
    cholesky_matrix: np.ndarray,
    time_step: float,
) -> Dict[str, List[float]]:
    """
    Generates correlated random shocks for Monte Carlo simulations.

    Args:
        num_iterations: The number of simulation iterations.
        num_assets: The number of assets.
        volatilities: An array of volatilities for each asset.
        cholesky_matrix: The Cholesky decomposition of the correlation matrix.
        time_step: The time step size.

    Returns:
        A dictionary where keys are asset names (e.g., "asset_1", "asset_2") and
        values are lists of generated shocks for each asset.
    """

    asset_names = [f"asset_{i + 1}" for i in range(num_assets)]
    random_normals = np.random.standard_normal(size=(num_iterations, num_assets))
    correlated_shocks = np.dot(random_normals, cholesky_matrix)
    shocks = volatilities * correlated_shocks * np.sqrt(time_step)
    shocks_df = pd.DataFrame(shocks, columns=asset_names)
    shocks_dict = shocks_df.to_dict(orient="list")
    return shocks_dict