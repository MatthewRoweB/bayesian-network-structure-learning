import numpy as np
import pandas as pd
from typing import Sequence, Union
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

 


def slarac_algorithm(temporal_data: pd.DataFrame, 
                    max_lags: int,
                    bootstrap_samples: int,
                    bootstrap_sample_sizes: Sequence[int],
                    *,
                    random_state: None | int = None,
                    weighted_lags_prob: Sequence[float] = None,
                    weighted_samples_prob: Sequence[float] = None) -> NDArray[float]:
    
    if len(bootstrap_sample_sizes) != bootstrap_samples:
        raise ValueError(f"There must be exactly {bootstrap_samples} sample sizes provided for boostrap_sample_sizes")
    
    data = temporal_data.to_numpy()
    t, d = data.shape
    
    A_full = np.zeros((d, d*max_lags), dtype=float)
    rng = np.random.default_rng(random_state)
    all_lags = np.arange(1, max_lags+1, 1, dtype=int)
    
    for v_b in range(bootstrap_samples):
        
        L = rng.choice(all_lags, p=weighted_lags_prob)
        
        eligible = np.arange(L, t)
        indices_sampled = rng.choice(eligible, 
                                     size=int(bootstrap_sample_sizes[v_b]), 
                                     replace=True, p=weighted_samples_prob)
        
        y_b = data[indices_sampled]
        x_b = np.concatenate([data[indices_sampled-i] for i in range(1, L+1)], axis=1)  # stack x's horizontally from lags 1 to L
        
        linear_regression_model = LinearRegression(fit_intercept=False)
        linear_regression_model.fit(x_b, y_b)                     # OLS estimate
        beta = np.abs(linear_regression_model.coef_)
        
        zero_pad_beta = np.pad(beta, ((0,0), (0, d*(max_lags-L))), mode='constant', constant_values=0)
        A_full += zero_pad_beta
           
    aggregate_betas = np.split(A_full, max_lags, axis=1)    
    A_score_matrix = np.maximum.reduce(aggregate_betas)           # max elementwise all zero-padded beta matrices
    return A_score_matrix


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\mattr\Downloads\slarac_sample_timeseries.csv")        
    A_score = slarac_algorihm(df, max_lags=2, bootstrap_samples=5, bootstrap_sample_sizes=np.array([4,5,7,8,10])) 
    n, p = df.shape
    assert A_score.shape == (p,p)
    print(A_score)
    
    





































