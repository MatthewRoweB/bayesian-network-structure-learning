import numpy as np
import pyvinecopulib as pvc
import torch
import torch.nn as nn
from quantile_forest import RandomForestQuantileRegressor
from quantnn import QRNN
from dataclasses import dataclass, field
from sklearn.preprocessing import QuantileTransformer
from typing import Literal, Tuple, Sequence, Any, Dict, NamedTuple, Protocol, runtime_checkable

@runtime_checkable
class QuantileRegression(Protocol):
    def predict(self, X: np.ndarray, Y: np.ndarray, quantiles: Sequence[float]) -> None:
        pass
        
@dataclass
class RandomForestQuantileRegression:
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    predict_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def predict(self, X: np.ndarray, Y: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
        model = RandomForestQuantileRegressor(**self.init_kwargs)
        model.fit(X, Y, sample_weight=None)     # sample weight is None as it might skew the causal direction
        prediction = model.predict(X, quantiles=list(quantiles), **self.predict_kwargs)
        return prediction
    
@dataclass
class NeuralNetworkQuantileRegression: 
    nn_model: nn.Sequential = None              
    train_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_loader: Dict[str, Any] = field(default_factory=dict)
    
    def _build_model(self, quantiles):
        if self.nn_model is None: 
            n_layers = 4
            n_neurons = 256
    
            layers = [
                nn.Linear(1, n_neurons),     
                nn.BatchNorm1d(n_neurons),
                nn.ReLU()
            ]
    
            for _ in range(n_layers):
                layers.extend([
                    nn.Linear(n_neurons, n_neurons),
                    nn.BatchNorm1d(n_neurons),
                    nn.ReLU()
                ])
    
            layers.append(nn.Linear(n_neurons, len(quantiles)))
            self.nn_model = nn.Sequential(*layers)
            
        return self.nn_model 
            
    def predict(self, X: np.ndarray, Y: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
        from torch.utils.data import TensorDataset, DataLoader

        model = self._build_model(quantiles)
        qrnn = QRNN(quantiles=quantiles, model=model)

        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(Y, dtype=torch.float32).ravel()

        training_data = TensorDataset(x_tensor, y_tensor)
        training_loader = DataLoader(training_data, **self.data_loader)
    
        # --- build optimizer / scheduler here ---
        opt_cls = self.train_kwargs.get("optimizer_cls", torch.optim.SGD)
        opt_kwargs = self.train_kwargs.get("optimizer_kwargs", {"lr": 0.1, "momentum": 0.9})
        optimizer = opt_cls(model.parameters(), **opt_kwargs)

        sched_cls = self.train_kwargs.get("scheduler_cls")
        sched_kwargs = self.train_kwargs.get("scheduler_kwargs", {})
        n_epochs = self.train_kwargs.get("n_epochs", 2)

        if sched_cls is not None:
            scheduler = sched_cls(optimizer, **sched_kwargs)
        else:
            scheduler = None

        if scheduler is None:
            qrnn.train(training_loader, optimizer=optimizer, n_epochs=n_epochs)
        else:
            qrnn.train(training_loader, optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs)

        prediction = qrnn.predict(x_tensor).numpy()
        return prediction
    
@dataclass
class BivariateCopula:
    bivariate_copula_model: pvc.Bicop 
    
    @staticmethod
    def rank_transform(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.size
        tx = QuantileTransformer(output_distribution='uniform', n_quantiles=n)
        ty = QuantileTransformer(output_distribution='uniform', n_quantiles=n)
        X=tx.fit_transform(X)
        Y=ty.fit_transform(Y)
        return X, Y
    
    
    ####################
    @staticmethod
    def rank_inverse(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.size
        tx = QuantileTransformer(output_distribution='uniform', n_quantiles=n)
        ty = QuantileTransformer(output_distribution='uniform', n_quantiles=n)
        X=tx.fit_transform(X)
        Y=ty.fit_transform(Y)
        return X, Y
    
    
    
    def predict(self, X: np.ndarray, Y: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
        X_ranked, Y_ranked = self.rank_transform(X, Y)
        
        
        
        
        
        pass
    
    
    
    

class CausalDirAndScore(NamedTuple):
    
    direction: str
    confidence_score: float


def quadrature_weights_and_nodes(quadrature: Literal['legendre', 'uniform'], 
                                m: int, 
                                lower_quantile: float, 
                                upper_quantile: float
                                ) -> Tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    
    a = float(lower_quantile)
    b = float(upper_quantile)

    if quadrature == 'legendre':
        # nodes in [-1,1], weights for that domain
        nodes, w = np.polynomial.legendre.leggauss(m)
        # affine map to [a,b]
        taus = (b - a) * (nodes + 1.0) / 2.0 + a
        weights = w * (b - a) / 2.0
    else:   # Uniform Quadrature
        # simple rectangle rule on [a,b]
        taus = np.linspace(a, b, m, endpoint=True)
        # equal weights summing to (b-a)
        weights = np.full(m, (b - a) / m)  
    return taus, weights                  # list(taus) as some predict() methods for qr accept lists but not arrays
    

def empirical_quantile(X: np.ndarray,
                       Y: np.ndarray,
                       taus: np.ndarray):
    
    # Empirical marginal quantiles
    q_hat_X = np.quantile(X, q=taus, axis=0, method='inverted_cdf')    
    q_hat_Y = np.quantile(Y, q=taus, axis=0, method='inverted_cdf')

    I_x = (X.T <= q_hat_X)       # indicators 1[x <= q] and 1[y <= q]
    I_y = (Y.T <= q_hat_Y)
    
    # pinball loss for all taus
    S_hat_X = ((taus[:,np.newaxis] - I_x) * (X.T - q_hat_X)).sum(axis=1)      # (k,)
    S_hat_Y = ((taus[:,np.newaxis] - I_y) * (Y.T - q_hat_Y)).sum(axis=1)      # (k,)

    return S_hat_X, S_hat_Y


def quantile_regression(X: np.ndarray,
                        Y: np.ndarray,
                        taus: list[np.float64],
                        qr: RandomForestQuantileRegression | NeuralNetworkQuantileRegression | BivariateCopula,
                        ) -> Tuple[np.ndarray, np.ndarray]:
  
    q_hat_y_given_x = qr.predict(X, Y.ravel(), quantiles=taus)   # Y | X
    q_hat_x_given_y = qr.predict(Y, X.ravel(), quantiles=taus)   # X | Y  
    
    I_x = (X <= q_hat_x_given_y).astype(float)  # (n, m)   
    I_y = (Y <= q_hat_y_given_x).astype(float)  # (n, m)
    
    S_hat_X_given_Y = np.sum((taus[np.newaxis, :] - I_x) * (X - q_hat_x_given_y), axis=0)  # X|Y   # (k,)
    S_hat_Y_given_X = np.sum((taus[np.newaxis, :] - I_y) * (Y - q_hat_y_given_x), axis=0)  # Y|X   # (k,)
    
    return S_hat_X_given_Y, S_hat_Y_given_X


def bivariate_quantile_causal_discovery(X: np.ndarray,
                                        Y: np.ndarray, 
                                        qr: RandomForestQuantileRegression | 
                                            NeuralNetworkQuantileRegression | 
                                            BivariateCopula , # TODO: add copulas
                                        m: int,                       
                                        lower_quantile: float, 
                                        upper_quantile: float,
                                        *,
                                        quadrature: Literal['legendre', 'uniform'] = 'legendre', 
                                        random_state: int | None = None
                                        ) -> CausalDirAndScore:
        
    if (n_x := X.shape[0]) != (n_y := Y.shape[0]):
        raise ValueError(f"X and Y must both have same number of samples. Got unmatched samples {n_x} and {n_y}")
        
    if not isinstance(m, int):
        raise ValueError(f"The number of quantile levels m must be a positive integer. Got {m}")
        
    if not (0 < lower_quantile < 1 and 0 < upper_quantile < 1):
        raise ValueError(f"Lower quantile and upper quantile must both between 0 and 1. Got {lower_quantile, upper_quantile}")
        
    if (X.ndim != 2 or X.shape[1] != 1) or (Y.ndim != 2 or Y.shape[1] != 1):
        raise ValueError(f"Expected 2D arrays of shape (n,1) for both X and Y. Got {X.shape} and {Y.shape}. Reshape your \
                         arrays using array.reshape(-1,1)")
        
    if not isinstance(qr, QuantileRegression):
        raise ValueError("""The quantile regression method qr must be a model instantiated from a class using the
                          Quantile Regression Protocol which can be of type: RandomForestQuantileRegressor, 
                          NeuralNetworkQuantileRegressor, NonParametricBivariateCopula""" )
    
    tx = QuantileTransformer(output_distribution='normal', n_quantiles=n_x)  # Transformation to N(0,1) 
    ty = QuantileTransformer(output_distribution='normal', n_quantiles=n_x)  # Resolution discretization to sample size n 
    X=tx.fit_transform(X)
    Y=ty.fit_transform(Y)
    
    taus, weights = quadrature_weights_and_nodes(quadrature, m, lower_quantile, upper_quantile)
    
    S_X_tau, S_Y_tau = empirical_quantile(X, Y, taus)
    S_X_given_Y_tau, S_Y_given_X_tau = quantile_regression(X, Y, taus, qr)

    score_x_y = weights @ S_X_tau + weights @ S_Y_given_X_tau    # Sx + Sy|x
    score_y_x = weights @ S_Y_tau + weights @ S_X_given_Y_tau    # Sy + Sx|y
    
    if score_x_y < score_y_x:
        # X->Y is better (smaller risk)
        direction = "X->Y"
        confidence = score_y_x - score_x_y   # non-negative
    else:
        direction = "Y->X"
        confidence = score_x_y - score_y_x   # non-negative

    causal_score = CausalDirAndScore(direction, float(confidence))
        
    return causal_score


if __name__ == "__main__":
    
    def generate_bivariate_data(n: int = 500, seed: int = 0):
        rng = np.random.default_rng(seed)
        
        # X ~ Uniform + small Gaussian ripple
        X = rng.uniform(-3, 3, size=n)
        
        # Nonlinear and heteroskedastic relationship:
        # Y = sin(X) + X/2 + (noise scale depends on |X|)
        noise = rng.normal(0, 0.5 + 0.3 * np.abs(X))
        Y = np.sin(X) + 0.5 * X + noise
        
        # reshape to 2D for sklearn / quantile_forest
        return X.reshape(-1,1), Y.reshape(-1,1) 


    X_data, Y_data = generate_bivariate_data(n=1000)

    model = NeuralNetworkQuantileRegression(
        train_kwargs={
            "optimizer_cls": torch.optim.SGD,
            "optimizer_kwargs": {"lr": 0.1, "momentum": 0.9},
            "scheduler_cls": torch.optim.lr_scheduler.CosineAnnealingLR,
            "scheduler_kwargs": {"T_max": 2},   # or n_epochs, etc.
            "n_epochs": 2,
        },
        data_loader={"batch_size": 256, "shuffle": True, "num_workers": 0},
    )

    result_nn = bivariate_quantile_causal_discovery(X_data, Y_data, model, 10, 0.05, 0.95)
    print(result_nn)

    rf_qr = RandomForestQuantileRegression(
        init_kwargs={"n_estimators": 200, "min_samples_leaf": 5}
    )

    result_rf = bivariate_quantile_causal_discovery(
        X_data, Y_data,
        qr=rf_qr,
        m=15,
        lower_quantile=0.05,
        upper_quantile=0.95,
    )
    print(result_rf)



















