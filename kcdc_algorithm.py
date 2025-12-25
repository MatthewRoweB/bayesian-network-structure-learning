from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
from typing import Protocol, Literal, ClassVar, Dict, Callable, Tuple, runtime_checkable, NamedTuple, Optional, get_args
from dataclasses import dataclass, field
from itertools import combinations, count, batched
from scipy.linalg import cho_factor, cho_solve, LinAlgError

@runtime_checkable
class Classification(Protocol):
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass 
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:      # confidence_scores needed for classification 
        pass

KernelName = Literal["rbf", "laplace", "log", "polynomial", "cauchy", "sigmoid", "linear", "matern", "rational_quadratic"]
 
@dataclass(frozen=True, slots=True)
class Kernel:
    kernel: KernelName 
    parameters: Dict[str, float] = field(default_factory=dict) # {gamma: float, intercept: float, degree: int, alpha: float, length_scale: float}
    
    def __post_init__(self):
        if self.kernel not in get_args(KernelName):
            raise ValueError(f"{self.kernel} is not recognized, must be one of {KernelName}")
            
        params_check = set(self.parameters.keys())
        possible_params = {'gamma', 'q', 'intercept', 'degree', 'alpha', 'coef0', 'length_scale', 'nu'}
        if params_check.issubset(possible_params) is False:
            raise ValueError(f"At least one parameter from {params_check} is not recognized, must be of \
                             {possible_params}")
                             
    def __hash__(self):
        return hash(tuple((self.kernel, tuple(self.parameters.items()))))  # caching kernels 
     
    @staticmethod
    def rbf(X, gamma=1.0):
        k = pairwise.rbf_kernel(X, None, gamma)
        return k 
    
    @staticmethod
    def laplace(X, gamma=1.0):
        k = pairwise.laplacian_kernel(X, None, gamma)
        return k 
        
    # TODO: implement log kernel
    @staticmethod
    def log(X):
        pass
        
    @staticmethod
    def linear(X):
        k = pairwise.linear_kernel(X, None)
        return k
    
    @staticmethod
    def polynomial(X, degree=3, gamma=1.0, coef0=1):
       k =  pairwise.polynomial_kernel(X, None, degree, gamma, coef0)
       return k
    
    @staticmethod
    def cauchy(X, gamma=1.0):
        X = np.ravel(X)
        dist2 = (np.subtract.outer(X, X) ** 2)
        k = 1.0 / (1.0 + dist2 / gamma)
        return k
    
    @staticmethod
    def sigmoid(X, gamma=1.0, intercept=0.0):
        X = np.ravel(X)
        k = np.tanh(gamma * np.add.outer(X,X) + intercept)
        return k 
    
    _kernel_choice: ClassVar[Dict[KernelName, Callable]] = {
        "rbf": rbf,
        'laplace': laplace,
        "log": log,
        "linear": linear,
        "polynomial": polynomial,
        "cauchy": cauchy,
        "sigmoid": sigmoid,
        #"rational_quadratic": rational_quadratic,
        #"matern": matern
        }
    
    cache_kernel_matrices: ClassVar = {}   # persists between calls
        
    def __call__(self, X, cache_id: Tuple[Kernel, str] | Tuple[None, None]):   
        
        x_name, y_name = cache_id    
        if (x_name is None and y_name is None):       # Z set and synthetic dataset no caching is possible
            X_scaled = StandardScaler().fit_transform(X)    
            gram_matrix = self._kernel_choice[self.kernel](X_scaled, **self.parameters)
            
        else:
            if cache_id in self.cache_kernel_matrices:
                return self.cache_kernel_matrices[cache_id]
            else:
                X_scaled = StandardScaler().fit_transform(X)    
                gram_matrix = self._kernel_choice[self.kernel](X_scaled, **self.parameters)

        return gram_matrix


def kernel_ridge(K: np.ndarray, regularization: float, I: np.ndarray) -> np.ndarray:
    
    try:
        M = K + regularization*I
        c, lower = cho_factor(M, overwrite_a=False, check_finite=False)
        return cho_solve((c, lower), K, check_finite=False)
    except LinAlgError:
        return np.linalg.pinv(M) @ K
         

def rkhs_norm_deviance(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                       kernels_xyz: Tuple[Kernel, ...], regularization: float,
                       cache_id: Optional[Tuple[str, str]] = None) -> Tuple[float, float]:
    
    if cache_id:
        x_name, y_name = cache_id
    else: 
        x_name, y_name = None, None
    
    if z.size==0:       # no conditioning set for two variables
        
        kernel_x, kernel_y = kernels_xyz[0], kernels_xyz[1]
        
        Lx = kernel_x(x, tuple((kernel_x, x_name)))
        Ly = kernel_y(y, tuple((kernel_x, y_name)))
        
        n = Lx.shape[1]
        I = np.eye(n, dtype=int)
        
        cme_y_x = max(float(np.trace(Lx)), 0.0)       # CME for Y | X 
        cme_x_y = max(float(np.trace(Ly)), 0.0)       # CME for X | Y 
        
        return cme_y_x, cme_x_y       # traces >= 0 since the squared norm in a RKHS >=0 

    else:
        kernel_x, kernel_y, kernel_z = kernels_xyz
        
        Lx = kernel_x(x, tuple((kernel_x, x_name)))
        Ly = kernel_y(y, tuple((kernel_y, y_name)))
        Lz = kernel_z(z, tuple((kernel_y, y_name)))
        
        n = Lz.shape[1]
        I = np.eye(n, dtype=int)
        Az =  kernel_ridge(Lz, regularization, I)
        
        Gy = Az.T @ Ly @ Az
        cme_y_x = max(float(np.trace(Gy)), 0.0)       # CME for Y | X  (but now Y | Z / {X})
        
        Gx = Az.T @ Lx @ Az
        cme_x_y = max(float(np.trace(Gx)), 0.0)       # CME for X | Y  (X | Z / {Y})
        
        return cme_y_x, cme_x_y


MechanismFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
GeneratorFunc = Callable[[np.random.Generator, Tuple[int, int]], np.ndarray]

class StructuralCausalModel:
    
    def __init__(self, mechanism: MechanismFunc, seed: np.random.Generator, 
                 x_dist: GeneratorFunc, y_dist: GeneratorFunc, z_dist: GeneratorFunc, 
                 eps_x_dist: GeneratorFunc, eps_y_dist: GeneratorFunc):
        
        self.mechanism = mechanism
        self.seed = seed
        self._validate_seed()                
        
        self.X = x_dist(self.seed)
        self.Y = y_dist(self.seed)
        self.Z = z_dist(self.seed)
        self.EPS_X = eps_x_dist(self.seed)
        self.EPS_Y = eps_y_dist(self.seed)
        
        self._validate_sample_size()
        
    def _validate_seed(self):
        if not isinstance(self.seed, np.random.Generator):
            raise TypeError(f"Seed must be of instance np.random.Generator by using np.random.default_rng(float), \
                             given seed: {self.seed}")
                             
    def _validate_sample_size(self):  
        common_shape = self.X.shape     # (n_obs, n_samples)
        if any(arr.shape != common_shape for arr in [self.X, self.Y, self.Z, self.EPS_X, self.EPS_Y]):
            raise ValueError("At least one of the distributions from x_dist, y_dist, z_dist, eps_x_dist, eps_y_dist \
                             has a shape mismatch. They all must share the same shape.")
    
    @property
    def x_y(self):    # X causes Y true direction
        self._x_y = self.mechanism(self.X, self.Z, self.EPS_Y)  # y = f(x) + g(z) + e
        return self._x_y
    
    @property
    def y_x(self):    # Y causes X true direction
        self._y_x = self.mechanism(self.Y, self.Z, self.EPS_X)  # asymmetry x = f(y) + g(z) + e
        return self._y_x
        
    def _generate_synthetic_dataset(self, kernels: list[Tuple[Kernel, ...]], regularization: float):
        
        n_samples = self.y_x.shape[1] 
        n_columns = len(list(kernels)) * 2
        
        X_training_matrix = np.empty(shape=(n_samples*2, n_columns), dtype=float)
        y_training = np.empty(shape=(n_samples*2,), dtype=float)
        
        row_counter = count(step=2)
        
        for j in range(n_samples):
            
            xj =   self.X[:, [j]]     # both xj and x_yj (n_obs, n_samples)
            x_yj = self.x_y[:, [j]]
            
            yj =   self.Y[:, [j]]     
            y_xj = self.y_x[:, [j]]                   
  
            zj =   self.Z[:, [j]]
            
            scores_class_1 = []
            scores_class_0 = []
            
            for kernel in zip(*kernels):
            
                # true causal direction X->Y (class 1)
                Sy_x1, Sx_y1 = rkhs_norm_deviance(xj, x_yj, zj, kernel, regularization) 
                # true causal direction Y->X (class 0)
                Sy_x0, Sx_y0 = rkhs_norm_deviance(yj, y_xj, zj, kernel, regularization) 
                
                scores_class_1.extend([Sy_x1, Sx_y1])
                scores_class_0.extend([Sy_x0, Sx_y0])
                    
            row = next(row_counter)
            
            X_training_matrix[row, :]   = scores_class_1
            X_training_matrix[row+1, :] = scores_class_0
                
            y_training[row]   = 1
            y_training[row+1] = 0           # y_training has 50/50 split for classes 1 and 0 well-balanced
            
        return X_training_matrix, y_training        
    
class CausalDirection(NamedTuple):
    direction: str
    confidence_score: float
    
# Kernel Conditional Deviance for Causal Inference 
def kcdc_algorithm(data: pd.DataFrame,
                   decision_rule: Literal['OneShot', 'MajorityVote', 'Classifier'] = 'OneShot', 
                   kernel_x: list[Kernel] = None,
                   kernel_y: list[Kernel] = None,
                   kernel_z: list[Kernel] = None,
                   *,
                   model: Classification = None,
                   scm: StructuralCausalModel = None,
                   regularization: float = 1e-1,
                   tiebreaker: float = 1e-3
                   ) -> list[CausalDirection]:

    kernel_x = [Kernel('rbf', {'gamma': 1})] if kernel_x is None else kernel_x     # defaults to only one hyperparamter
    kernel_y = [Kernel('rbf', {'gamma': 1})] if kernel_y is None else kernel_y     
    
    d = data.shape[1]
    
    if d < 2:
        raise ValueError("Dataset must contain at least 2 variables.")
    
    len_x, len_y, len_z = len(kernel_x), len(kernel_y), None if kernel_z is None else len(kernel_z)
    if d == 2:
        if len_x != len_y:
            raise ValueError(f"For d=2, kernel_x and kernel_y must have the same length. "
                             f"Got len(kernel_x)={len_x}, len(kernel_y)={len_y}.")
    
    elif d >= 3:
        if kernel_z is None:
            raise ValueError("For d>=3, kernel_z must be provided for conditioning.")
    
        if not (len_x == len_y == len_z):
            raise ValueError(f"For d>=3, kernel_x, kernel_y, and kernel_z must have the same length. "
                             f"Got len(kernel_x)={len_x}, len(kernel_y)={len_y}, len(kernel_z)={len_z}.")
    
    if decision_rule not in ['OneShot', 'MajorityVote', 'Classifier']:
        raise ValueError(f"The decision_rule: {decision_rule} does not exist. Must be of: OneShot, \
                         MajorityVote, or Classifier")
    
    if decision_rule == 'OneShot':
        if any(len_ker > 1 for len_ker in [len_x, len_y, len_z]):
            raise ValueError(f"OneShot uses only one kernel for each pair of variables. Kernels of length: \
                             kernel_x: {len_x}, kernel_y: {len_y}, kernel_z: {len_z} were passed")        
    
    if decision_rule == 'Classifier':
        if model is None: 
            model = LogisticRegression()     
        if not isinstance(model, Classification):
            raise TypeError(f"model must satisfy the Classification Protocol, given {model}")  
        if scm is None:
            raise ValueError("For Classifier decision rule a scm must be provided")
        if not isinstance(scm, StructuralCausalModel):
            raise TypeError(f"scm: {scm} must be of instance StructuralCausalModel")          
    
    Xn = data.to_numpy()
    colnames = list(data.columns)
    d = Xn.shape[1]
    
    causal_dir_names = []
    causal_dir_scores = np.empty(shape=((d*(d-1))//2, len_x*2), dtype=float)   # (n_samples, n_kernels_dir)
    
    row_counter = count(step=1)
    for i,j in combinations(range(d), 2):
        x_name = colnames[i]
        y_name = colnames[j]
        cache_id = tuple((x_name, y_name))
        causal_dir_names.extend([f"{y_name}->{x_name}", f"{x_name}->{y_name}"])
        
        x = Xn[:, [i]]
        y = Xn[:, [j]]
        z_indices = [k for k in range(d) if k not in (i,j)]
        z = Xn[:, z_indices]
        
        scores = []
        for kernels_xyz in zip(kernel_x, kernel_y, kernel_z):
            
            Sy_x, Sx_y = rkhs_norm_deviance(x, y, z, kernels_xyz, regularization, cache_id)
            
            print("Sy_x:", Sy_x, "Sx_y:", Sx_y)
            if abs(Sy_x - Sx_y) < tiebreaker:
                scores.extend([0, 0])   # Classifier takes in 0 columns for predict()
                continue
            
            scores.extend([Sy_x, Sx_y])    
            
        row = next(row_counter)
        causal_dir_scores[row, :] = scores
        
    causal_dir: list[CausalDirection] = []
    match decision_rule:
        case "OneShot":
            causal_dir_scores = causal_dir_scores[:,np.any(causal_dir_scores != 0, axis=0)]  # remove tiebreakers
            diff = causal_dir_scores[:, ::2] - causal_dir_scores[:, 1::2]
            confidence_scores = np.abs(diff) / np.minimum(causal_dir_scores[:, ::2]  + 1e-12, 
                                                          causal_dir_scores[:, 1::2] + 1e-12)
            diff_mask = np.where(diff > 0, 0, 1)
            
            for row, both_dir in enumerate(batched(causal_dir_names, 2)):
                label = diff_mask[row, 0]
                direction = both_dir[label]
                score = float(confidence_scores[row, 0])
                causal_dir.append(CausalDirection(direction, score))
                
            return causal_dir

        case "MajorityVote":        
            causal_dir_scores = causal_dir_scores[:,np.any(causal_dir_scores != 0, axis=0)]  # remove tiebreakers
            diff = causal_dir_scores[:, ::2] - causal_dir_scores[:, 1::2]
            diff_mask = np.where(diff > 0, 0, 1)
            
            diff_mask_majority = np.apply_along_axis(lambda arr: np.argmax(np.bincount(arr)), 
                                                     axis=1, arr=diff_mask)
            
            def majority_count(arr):
                
                if arr.size == 0:
                    return 0
                unique_values, counts = np.unique(arr, return_counts=True)
                max_count = counts.max()
                total_elements = arr.size
                ratio = max_count / total_elements
                return ratio
    
            majority_count = np.apply_along_axis(majority_count, axis=1, arr=diff_mask)
        
            for row, both_dir in enumerate(batched(causal_dir_names, 2)):
                label = diff_mask_majority[row]
                direction = both_dir[label]
                score = float(majority_count[row])
                causal_dir.append(CausalDirection(direction, score))
                
            return causal_dir
        
        case "Classifier": 
            X, y = scm._generate_synthetic_dataset([kernel_x, kernel_y, kernel_z], regularization)
            model.fit(X, y)
            probabilities = model.predict_proba(causal_dir_scores)
            
            predicted_labels = np.argmax(probabilities, axis=1)
            
            confidence_scores = np.take_along_axis(probabilities, predicted_labels.reshape(-1,1), axis=1).ravel()
            
            for row, both_dir in enumerate(batched(causal_dir_names, 2)):
                label = predicted_labels[row]
                direction = both_dir[label]
                score = float(confidence_scores[row])
                causal_dir.append(CausalDirection(direction, score))
            
            return causal_dir
    
if __name__ == '__main__':
    
    rng = np.random.default_rng(42)
    n = 500
    X = np.zeros((n, 10))
    
    # Base cause
    X[:, 0] = rng.normal(0, 1, size=n)  # X1
    
    # Causal branches from X1
    X[:, 1] = np.sin(X[:, 0]) + 0.5 * rng.normal(size=n)          # X2 = f(X1)
    X[:, 2] = np.log1p(X[:, 0]**2) + 0.5 * rng.normal(size=n)     # X3 = f(X1)
    
    # Causal chain from X3
    X[:, 3] = 0.3 * X[:, 2]**2 + 0.3 * rng.normal(size=n)         # X4 = f(X3)
    X[:, 4] = np.exp(0.1 * X[:, 3]) + 0.3 * rng.normal(size=n)    # X5 = f(X4)
    
    # Independent variable
    X[:, 5] = rng.normal(0, 1, size=n)                            # X6 (independent)
    
    # Mixed dependency
    X[:, 6] = 0.6 * X[:, 0] + 0.4 * X[:, 5] + rng.normal(size=n)  # X7 = f(X1, X6)
    X[:, 7] = X[:, 6]**2 + 0.5 * rng.normal(size=n)              # X8 = f(X7)
    
    # Downstream from earlier
    X[:, 8] = X[:, 4] + X[:, 7] + 0.5 * rng.normal(size=n)        # X9 = f(X5, X8)
    X[:, 9] = np.tanh(X[:, 8]) + 0.5 * rng.normal(size=n)         # X10 = f(X9)
    
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(10)])
    
    # ---- Kernels ---- #
    kern_x = [
    Kernel('rbf', {'gamma': 0.5}),
    Kernel('polynomial', {'degree': 2, 'gamma': 0.5, 'coef0': 1}),
    Kernel('linear', {})
    ]
    kern_y = [
    Kernel('laplace', {'gamma': 0.7}),
    Kernel('cauchy', {'gamma': 0.8}),
    Kernel('sigmoid', {'gamma': 1.0, 'intercept': 0})
    ]
    kern_z = [
    Kernel('laplace', {'gamma': 0.1}),
    Kernel('rbf', {'gamma': 0.1}),
    Kernel('linear', {})
    ]
    
    
    # ---- SCM for Training ---- #
    scm = StructuralCausalModel(
    mechanism=lambda x, z, eps: np.exp(0.3*x) + np.sin(2*x) + np.sin(0.5*z) + eps,
    seed=np.random.default_rng(42),
    x_dist=lambda rng: rng.normal(0, 1, (100, 1000)),
    y_dist=lambda rng: rng.normal(0, 1, (100, 1000)),
    z_dist=lambda rng: rng.normal(0, 1, (100, 1000)),
    eps_x_dist=lambda rng: rng.normal(0, 0.5, (100, 1000)),
    eps_y_dist=lambda rng: rng.normal(0, 0.5, (100, 1000))
    )
    
    # ---- Run KCDC ---- #
    import cProfile
    import pstats
    from pstats import SortKey
    
    with cProfile.Profile() as pf:
        result = kcdc_algorithm(
            df,
            decision_rule='Classifier',
            kernel_x=kern_x,
            kernel_y=kern_y,
            kernel_z=kern_z,
            scm=scm,
            model=LogisticRegression(max_iter=1000)
        )
        
        
    stats = pstats.Stats(pf)
    stats.strip_dirs()
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(15)
    
    print(result)







