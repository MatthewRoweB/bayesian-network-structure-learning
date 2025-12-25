import pandas as pd 
import numpy as np
import graphlib
from typing import Dict, Iterable, Protocol, Self, List, runtime_checkable
from sklearn.gaussian_process import GaussianProcessRegressor




@runtime_checkable
class NonLinearRegression(Protocol):
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

@runtime_checkable
class IndependenceTest(Protocol):
    
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

graph_dict = Dict[str, Iterable[str]]
 
# Non Linear Additive Noise Model
def non_linear_anm(dags: List[graph_dict],
                   non_linear_regression: NonLinearRegression,
                   independence_test: IndependenceTest,
                   alpha: float = 0.05, 
                   verbose: bool = False
                   ) -> List[graph_dict]:
    
    if isinstance(non_linear_regression, NonLinearRegression):
        raise TypeError(f"non_linear_regression must implement the methods fit and predict as in {NonLinearRegression} Protocol")
    if isinstance(independence_test, IndependenceTest):
        raise TypeError(f"independence_test must implement a __call__ method like in {IndependenceTest} Protocol")
        
    # validation
    for dag in dags:
        graphlib.TopologicalSorter()
    
    
    
    
    
    pass







