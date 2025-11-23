import logging
from typing import List, Literal
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
from torch.optim import Adam
# import numba
# from numba import njit, prange
#from arch import arch_model
# import scipy

EPS = 1e-9

# Return a tensor filled with NaNs
def get_nan_tensor(shape, device, dtype):
    return torch.full(shape, torch.nan, dtype=dtype, device=device)

# ------------------------------- Element-wise Operations -----------------------------
# Compute x / y with division by zero handling
def div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(y) > EPS, torch.divide(x, y), torch.nan)

# Compute log(x) with log(0) handling
def log(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(x) > EPS, torch.log(torch.abs(x)), torch.nan)


# ------------------------------- Helper Functions -----------------------------
# Unfold tensor x with window size d
def _unfold(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(x.shape + (d,), device=x.device, dtype=x.dtype)
    res[d - 1:] = x.unfold(0, d, 1)
    return res

# Calculate mean while ignoring NaNs (dim=-1)
def _mean(x: torch.Tensor) -> torch.Tensor:
    return x.nanmean(dim=-1)

# Calculate skewness while ignoring NaNs (dim=-1)
def _skew(x: torch.Tensor) -> torch.Tensor:
    x_demeaned = x - x.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True))
    zscore = torch.pow(div(x_demeaned, x_std), 3)
    return _mean(zscore)

# ------------------------------- Cross-Sectional Operations -----------------------------
# Cross-sectional ranking, with option for percentage rank
def cs_rank(x: torch.Tensor, valid_mask: torch.Tensor = None, pct: bool = True) -> torch.Tensor:
    ranks = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    ranks[~valid_mask] = torch.finfo(x.dtype).max
    ranks = torch.argsort(torch.argsort(ranks, dim=-1), dim=-1)
    ranks = ranks.to(dtype=x.dtype) + 1
    if pct:
        valid_counts = valid_mask.sum(dim=-1, keepdim=True)
        ranks = ranks / valid_counts
    ranks[~valid_mask] = torch.nan
    return ranks

# ------------------------------- Time-Series Operations -----------------------------
# Calculate time-series delay with lag d
def ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d:] = x[:-d]
    return res

# Calculate time-series delta with lag d
def ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d:] = x[d:] - x[: -d]
    return res

# Calculate time-series percentage change
def ts_pct(x: torch.Tensor) -> torch.Tensor:
    # Initialize a tensor filled with NaNs (same shape and device as input)
    res = torch.full_like(x, torch.nan)

    # When the number of time steps >= 2, calculate the percentage change for the previous T-1 steps
    if x.size(0) >= 2:
        current = x[:-1]  # Current value: [T-1, N]
        next_val = x[1:]  # Next time step value: [T-1, N]
        pct_change = div(next_val - current, current)
        res[1:] = pct_change  # Keep the first position as NaN
    
    return res

# Calculate time-series standard deviation over rolling windows of size d
def ts_stddev(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demean, 2), dim=-1))
    res[d - 1:] = x_std
    return res

# Calculate time-series mean over rolling windows of size d
def ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(dimension=0, size=d, step=1)
    res[d - 1:] = unfolded.nanmean(dim=-1)
    return res

# Calculate time-series exponentially weighted moving average over rolling windows of size d
def ts_ewma(x:torch.Tensor, d:int,alpha=None)->torch.Tensor:
    alpha = 1- 2/(d+1) if alpha is None else alpha
    assert alpha > 0 and alpha < 1, "Alpha must be in (0, 1)."
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    res[0] = x[0]
    weight = (1-alpha)**torch.arange(d, 0, -1, device=x.device) # (d,) 
    x_unfold = x.unfold(0, d, 1)
    #print(x_unfold.shape, weight.shape)
    x_weighted = torch.nansum(x_unfold * weight, dim=-1) / torch.nansum(weight)
    #print(x_weighted.shape)
    res[d-1:] = x_weighted
    return res

# Calculate time-series linear decay over rolling windows of size d
def ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    w = torch.arange(1, d + 1, dtype=torch.float32, device=x.device)
    w = w / w.sum()
    x_unfolded = x.unfold(dimension=0, size=d, step=1)
    w = w.view(1, 1, -1)
    res = torch.nansum(x_unfolded * w, dim=-1)
    result = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    result[d - 1:] = res
    return result

# Calculate time-series skewness over rolling windows of size d
def ts_skew(x: torch.Tensor, d: int) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    return _skew(x_unfold)

# Calculate time-series regression beta over rolling windows of size d
def ts_regbeta(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    x_mean = x_unfold.nanmean(dim=-1, keepdim=True)
    y_mean = y_unfold.nanmean(dim=-1, keepdim=True)
    x_demean = x_unfold - x_mean
    y_demean = y_unfold - y_mean
    std = torch.nansum(torch.pow(x_demean, 2), dim=-1)
    res[d - 1:] = torch.where(std < EPS, torch.nan, torch.nansum(x_demean * y_demean, dim=-1) / std)
    return res

# ts_gmm_1stmean (optimized version)
def ts_gmm_1stmean(x: torch.Tensor, d: int, em_iters: int = 10) -> torch.Tensor:
    """
    Use a sliding window to fit a 2-component GMM (based on the EM algorithm) 
    to the time series data of each stock, and return the long-term stable mean 
    (i.e., the mean of the first component).
    
    Parameters:
        x: Input tensor of shape [num_minutes, num_stocks]
        d: Sliding window size
        em_iters: Number of EM algorithm iterations (default is 10, can be adjusted as needed)
    
    Returns:
        res: A tensor with the same shape as x, where only the last row of each window 
             contains the GMM result for that window, and all other values are NaN.
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0] < d:
        return res

    # 1. Extract sliding windows using unfold, shape will be [num_windows, num_stocks, d]
    windows = x.unfold(dimension=0, size=d, step=1)
    num_windows, n_stocks, _ = windows.shape
    B = num_windows * n_stocks
    # Reshape windows to 2D: each row represents d time points for a stock in a window
    windows = windows.reshape(B, d)  # shape: [B, d]

    # 2. Handle NaN values on GPU: for each window, fill NaNs with the mean of that window
    # Compute the mean for each row (ignoring NaNs), NaN will be returned if all values are NaN
    window_means = torch.nanmean(windows, dim=1, keepdim=True)  # shape: [B, 1]
    windows_filled = torch.where(torch.isnan(windows), window_means, windows)  # [B, d]

    # 3. Initialize parameters for the 2-component GMM (processing each window independently)
    # Initial means: the first component gets the median, the second component gets the 80th percentile
    medians = torch.median(windows_filled, dim=1).values  # shape: [B]
    # !! prior knowledge: the short-term price jump may be the 80th percentile, which can be adjusted
    perc80 = torch.quantile(windows_filled, 0.8, dim=1)     # shape: [B]
    var_init = torch.var(windows_filled, dim=1, unbiased=False)  # shape: [B]
    var_init = var_init.clamp(min=1e-6)  # Prevent zero variance

    # All parameters are of shape [B, 2]
    means = torch.stack([medians, perc80], dim=1) # Initial means
    variances = var_init.unsqueeze(1).repeat(1, 2) # Initial variances
    weights = torch.full((B, 2), 0.5, dtype=x.dtype, device=x.device) # Equal weights

    # Expand window data to [B, d, 1] for later broadcasting
    X = windows_filled.unsqueeze(2)  # shape: [B, d, 1]

    # 4. EM algorithm iterations (all parallelized on GPU)
    for _ in range(em_iters):
        # E-step: Compute posterior probabilities (responsibilities)
        # Expand means and variances for broadcasting
        mu = means.unsqueeze(1)         # [B, 1, 2]
        sigma2 = variances.unsqueeze(1)   # [B, 1, 2]
        # Compute the Gaussian distribution coefficients
        coef = 1.0 / torch.sqrt(2 * torch.pi * sigma2)  # [B, 1, 2]
        # Compute the exponential part
        exp_term = torch.exp(-0.5 * ((X - mu) ** 2) / sigma2)  # [B, d, 2]
        likelihood = coef * exp_term  # [B, d, 2]
        # Weight by the mixture weights, note that weights need to be expanded
        weighted_likelihood = weights.unsqueeze(1) * likelihood  # [B, d, 2]
        # Normalize by summing across the components, compute the denominator for posterior probabilities
        normalizer = weighted_likelihood.sum(dim=2, keepdim=True)  # [B, d, 1]
        # Compute responsibilities, add 1e-10 to avoid division by zero
        resp = weighted_likelihood / (normalizer + 1e-10)  # [B, d, 2]

        # M-step: Update parameters
        # Calculate the responsibility sums for each component: [B, 2]
        resp_sum = resp.sum(dim=1)
        # Update means: weighted average
        means = (resp * X).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        # Update variances: weighted variance
        variances = (resp * (X - means.unsqueeze(1))**2).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        # Update weights: average responsibility for each component
        weights = resp_sum / d  # [B, 2]

    # 5. Take the mean of the first component as the long-term stable mean
    first_component_mean = means[:, 0]  # [B]
    # Reshape to [num_windows, num_stocks]
    gmm_means = first_component_mean.reshape(num_windows, n_stocks)

    # 6. Place the computed result into res, with the last time point of each window filled with the GMM result
    res[d-1:, :] = gmm_means.to(x.device)
    return res

# ts_gmm_2ndmean (optimized version)
def ts_gmm_2ndmean(x: torch.Tensor, d: int, em_iters: int = 10) -> torch.Tensor:
    """
    Use a sliding window to fit a 2-component GMM (based on the EM algorithm) 
    to the time series data of each stock, and return the short-term price jump mean 
    (i.e., the mean of the second component).
    
    Parameters:
        x: Input tensor of shape [num_minutes, num_stocks]
        d: Sliding window size
        em_iters: Number of EM algorithm iterations (default is 10, can be adjusted as needed)
    
    Returns:
        res: A tensor with the same shape as x, where only the last row of each window 
             contains the GMM result for that window, and all other values are NaN.
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0] < d:
        return res

    # 1. Extract sliding windows using unfold, shape will be [num_windows, num_stocks, d]
    windows = x.unfold(dimension=0, size=d, step=1)
    num_windows, n_stocks, _ = windows.shape
    B = num_windows * n_stocks
    # Reshape windows to 2D: each row represents d time points for a stock in a window
    windows = windows.reshape(B, d)  # shape: [B, d]

    # 2. Handle NaN values on GPU: for each window, fill NaNs with the mean of that window
    # Compute the mean for each row (ignoring NaNs), NaN will be returned if all values are NaN
    window_means = torch.nanmean(windows, dim=1, keepdim=True)  # shape: [B, 1]
    windows_filled = torch.where(torch.isnan(windows), window_means, windows)  # [B, d]

    # 3. Initialize parameters for the 2-component GMM (processing each window independently)
    # Initial means: the first component gets the median, the second component gets the 80th percentile
    medians = torch.median(windows_filled, dim=1).values  # shape: [B]
    # !! prior knowledge: the short-term price jump may be the 80th percentile, which can be adjusted
    perc80 = torch.quantile(windows_filled, 0.8, dim=1)     # shape: [B]
    var_init = torch.var(windows_filled, dim=1, unbiased=False)  # shape: [B]
    var_init = var_init.clamp(min=1e-6)  # Prevent zero variance

    # All parameters are of shape [B, 2]
    means = torch.stack([medians, perc80], dim=1) # Initial means
    variances = var_init.unsqueeze(1).repeat(1, 2) # Initial variances
    weights = torch.full((B, 2), 0.5, dtype=x.dtype, device=x.device) # Equal weights

    # Expand window data to [B, d, 1] for later broadcasting
    X = windows_filled.unsqueeze(2)  # shape: [B, d, 1]

    # 4. EM algorithm iterations (all parallelized on GPU)
    for _ in range(em_iters):
        # E-step: Compute posterior probabilities (responsibilities)
        # Expand means and variances for broadcasting
        mu = means.unsqueeze(1)         # [B, 1, 2]
        sigma2 = variances.unsqueeze(1)   # [B, 1, 2]
        # Compute the Gaussian distribution coefficients
        coef = 1.0 / torch.sqrt(2 * torch.pi * sigma2)  # [B, 1, 2]
        # Compute the exponential part
        exp_term = torch.exp(-0.5 * ((X - mu) ** 2) / sigma2)  # [B, d, 2]
        likelihood = coef * exp_term  # [B, d, 2]
        # Weight by the mixture weights, note that weights need to be expanded
        weighted_likelihood = weights.unsqueeze(1) * likelihood  # [B, d, 2]
        # Normalize by summing across the components, compute the denominator for posterior probabilities
        normalizer = weighted_likelihood.sum(dim=2, keepdim=True)  # [B, d, 1]
        # Compute responsibilities, add 1e-10 to avoid division by zero
        resp = weighted_likelihood / (normalizer + 1e-10)  # [B, d, 2]

        # M-step: Update parameters
        # Calculate the responsibility sums for each component: [B, 2]
        resp_sum = resp.sum(dim=1)
        # Update means: weighted average
        means = (resp * X).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        # Update variances: weighted variance
        variances = (resp * (X - means.unsqueeze(1))**2).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        # Update weights: average responsibility for each component
        weights = resp_sum / d  # [B, 2]

    # 5. Take the mean of the second component as the short-term price jump mean
    second_component_mean = means[:, 1]  # [B]
    # Reshape to [num_windows, num_stocks]
    gmm_means = second_component_mean.reshape(num_windows, n_stocks)

    # 6. Place the computed result into res, with the last time point of each window filled with the GMM result
    res[d-1:, :] = gmm_means.to(x.device)
    return res

# Obtain the long-term stable standard deviation (optimized version)
def ts_gmm_1ststd(x: torch.Tensor, d: int, em_iters: int = 10) -> torch.Tensor:
    """
    Use a sliding window to fit a 2-component GMM (based on the EM algorithm) 
    to the time series data of each stock, and return the long-term stable standard deviation 
    (i.e., the standard deviation of the first component).
    
    Parameters:
        x: Input tensor of shape [num_minutes, num_stocks]
        d: Sliding window size
        em_iters: Number of EM algorithm iterations (default is 10, can be adjusted as needed)
    
    Returns:
        res: A tensor with the same shape as x, where only the last row of each window 
             contains the GMM result for that window, and all other values are NaN.
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0] < d:
        return res

    # 1. Extract sliding windows using unfold, shape will be [num_windows, num_stocks, d]
    windows = x.unfold(dimension=0, size=d, step=1)
    num_windows, n_stocks, _ = windows.shape
    B = num_windows * n_stocks
    # Reshape windows to 2D: each row represents d time points for a stock in a window
    windows = windows.reshape(B, d)  # shape: [B, d]

    # 2. Handle NaN values on GPU: for each window, fill NaNs with the mean of that window
    window_means = torch.nanmean(windows, dim=1, keepdim=True)  # shape: [B, 1]
    windows_filled = torch.where(torch.isnan(windows), window_means, windows)  # [B, d]

    # 3. Initialize parameters for the 2-component GMM (processing each window independently)
    medians = torch.median(windows_filled, dim=1).values  # shape: [B]
    perc80 = torch.quantile(windows_filled, 0.8, dim=1)     # shape: [B]
    var_init = torch.var(windows_filled, dim=1, unbiased=False)  # shape: [B]
    var_init = var_init.clamp(min=1e-6)  # Prevent zero variance

    # All parameters are of shape [B, 2]
    means = torch.stack([medians, perc80], dim=1)  # Initial means
    variances = var_init.unsqueeze(1).repeat(1, 2)  # Initial variances
    weights = torch.full((B, 2), 0.5, dtype=x.dtype, device=x.device)  # Equal weights

    # Expand window data to [B, d, 1] for later broadcasting
    X = windows_filled.unsqueeze(2)  # shape: [B, d, 1]

    # 4. EM algorithm iterations (all parallelized on GPU)
    for _ in range(em_iters):
        # E-step: Compute posterior probabilities (responsibilities)
        mu = means.unsqueeze(1)         # [B, 1, 2]
        sigma2 = variances.unsqueeze(1)   # [B, 1, 2]
        coef = 1.0 / torch.sqrt(2 * torch.pi * sigma2)  # [B, 1, 2]
        exp_term = torch.exp(-0.5 * ((X - mu) ** 2) / sigma2)  # [B, d, 2]
        likelihood = coef * exp_term  # [B, d, 2]
        weighted_likelihood = weights.unsqueeze(1) * likelihood  # [B, d, 2]
        normalizer = weighted_likelihood.sum(dim=2, keepdim=True)  # [B, d, 1]
        resp = weighted_likelihood / (normalizer + 1e-10)  # [B, d, 2]

        # M-step: Update parameters
        resp_sum = resp.sum(dim=1)
        means = (resp * X).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        variances = (resp * (X - means.unsqueeze(1))**2).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        weights = resp_sum / d  # [B, 2]

    # 5. Take the standard deviation of the first component as the long-term stable standard deviation
    first_component_std = torch.sqrt(variances[:, 0])  # [B]
    gmm_stds = first_component_std.reshape(num_windows, n_stocks)

    # 6. Place the computed result into res, with the last time point of each window filled with the GMM result
    res[d-1:, :] = gmm_stds.to(x.device)
    return res

# Obtain the short-term price jump standard deviation (optimized version)
def ts_gmm_2ndstd(x: torch.Tensor, d: int, em_iters: int = 10) -> torch.Tensor:
    """
    Use a sliding window to fit a 2-component GMM (based on the EM algorithm) 
    to the time series data of each stock, and return the short-term price jump standard deviation 
    (i.e., the standard deviation of the second component).
    
    Parameters:
        x: Input tensor of shape [num_minutes, num_stocks]
        d: Sliding window size
        em_iters: Number of EM algorithm iterations (default is 10, can be adjusted as needed)
    
    Returns:
        res: A tensor with the same shape as x, where only the last row of each window 
             contains the GMM result for that window, and all other values are NaN.
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0] < d:
        return res

    # 1. Extract sliding windows using unfold, shape will be [num_windows, num_stocks, d]
    windows = x.unfold(dimension=0, size=d, step=1)
    num_windows, n_stocks, _ = windows.shape
    B = num_windows * n_stocks
    # Reshape windows to 2D: each row represents d time points for a stock in a window
    windows = windows.reshape(B, d)  # shape: [B, d]

    # 2. Handle NaN values on GPU: for each window, fill NaNs with the mean of that window
    window_means = torch.nanmean(windows, dim=1, keepdim=True)  # shape: [B, 1]
    windows_filled = torch.where(torch.isnan(windows), window_means, windows)  # [B, d]

    # 3. Initialize parameters for the 2-component GMM (processing each window independently)
    medians = torch.median(windows_filled, dim=1).values  # shape: [B]
    perc80 = torch.quantile(windows_filled, 0.8, dim=1)     # shape: [B]
    var_init = torch.var(windows_filled, dim=1, unbiased=False)  # shape: [B]
    var_init = var_init.clamp(min=1e-6)  # Prevent zero variance

    # All parameters are of shape [B, 2]
    means = torch.stack([medians, perc80], dim=1)  # Initial means
    variances = var_init.unsqueeze(1).repeat(1, 2)  # Initial variances
    weights = torch.full((B, 2), 0.5, dtype=x.dtype, device=x.device)  # Equal weights

    # Expand window data to [B, d, 1] for later broadcasting
    X = windows_filled.unsqueeze(2)  # shape: [B, d, 1]

    # 4. EM algorithm iterations (all parallelized on GPU)
    for _ in range(em_iters):
        # E-step: Compute posterior probabilities (responsibilities)
        mu = means.unsqueeze(1)         # [B, 1, 2]
        sigma2 = variances.unsqueeze(1)   # [B, 1, 2]
        coef = 1.0 / torch.sqrt(2 * torch.pi * sigma2)  # [B, 1, 2]
        exp_term = torch.exp(-0.5 * ((X - mu) ** 2) / sigma2)  # [B, d, 2]
        likelihood = coef * exp_term  # [B, d, 2]
        weighted_likelihood = weights.unsqueeze(1) * likelihood  # [B, d, 2]
        normalizer = weighted_likelihood.sum(dim=2, keepdim=True)  # [B, d, 1]
        resp = weighted_likelihood / (normalizer + 1e-10)  # [B, d, 2]

        # M-step: Update parameters
        resp_sum = resp.sum(dim=1)
        means = (resp * X).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        variances = (resp * (X - means.unsqueeze(1))**2).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        weights = resp_sum / d  # [B, 2]

    # 5. Take the standard deviation of the second component as the short-term price jump standard deviation
    second_component_std = torch.sqrt(variances[:, 1])  # [B]
    gmm_stds = second_component_std.reshape(num_windows, n_stocks)

    # 6. Place the computed result into res, with the last time point of each window filled with the GMM result
    res[d-1:, :] = gmm_stds.to(x.device)
    return res


