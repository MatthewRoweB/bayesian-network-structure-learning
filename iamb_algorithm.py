import numpy as np
import pandas as pd
from scipy.stats import entropy
import logging

from pgmpy.estimators import CITests


# ---------------------------------- Logging ---------------------------------
logger = logging.getLogger("iamb_logger")
logger.setLevel(logging.INFO)

# This line is critical to prevent printing to console
logger.propagate = False

# Remove all existing handlers (just in case)
if logger.hasHandlers():
    logger.handlers.clear()

# Add only FileHandler — not StreamHandler
file_handler = logging.FileHandler("iamb_log.txt", mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


if not logger.hasHandlers():
    file_handler = logging.FileHandler("iamb_log.txt", mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)



# ------------------- Start of the Algorithm ---------------------------------

def max_variable_conditional_mutual_info(df: pd.DataFrame, 
                                         target: str,
                                         candidate_markov_blanket: list[str]) -> tuple[str, float]:
    def H(*cols):
        probs = df[list(cols)].value_counts(normalize=True)
        probs = probs[probs > 1e-12]
        return entropy(probs, base=np.e)

    variables = (col for col in df.columns if col != target and col not in candidate_markov_blanket)
    cmb = candidate_markov_blanket
    cmi_scores = {}

    for x in variables:
        if not cmb:
            cmi = H(x) + H(target) - H(x, target)
        else:
            cmi = H(x, *cmb) + H(target, *cmb) - H(x, target, *cmb) - H(*cmb)
        cmi_scores[x] = cmi

    if not cmi_scores:
        return None, 0

    best_var = max(cmi_scores, key=cmi_scores.get)
    return best_var, cmi_scores[best_var]


def growing_phase(df: pd.DataFrame,
                  target: str,
                  threshold: float = 1e-4, 
                  verbose: bool = False) -> set[str]:
    mb = set()
    while True:
        best_var, best_score = max_variable_conditional_mutual_info(df, target, list(mb))
        if best_var is None or best_score < threshold:
            break
        if verbose:
            logger.info(f"Added to MB: {best_var} (CMI={best_score:.12f})")
        mb.add(best_var)
    return mb


def shrinking_phase(df: pd.DataFrame,
                    target: str,
                    markov_blanket: set,
                    ci_test_name: str = "g_sq",
                    alpha: float = 0.05,
                    verbose: bool = False) -> set[str]:

    method_map = {
        "chi_square": CITests.chi_square,
        "g_sq": CITests.g_sq,
        "log_likelihood": CITests.log_likelihood,
        "modified_log_likelihood": CITests.modified_log_likelihood,
        "power_divergence": CITests.power_divergence,
        "pearsonr": CITests.pearsonr
    }

    if ci_test_name not in method_map:
        raise ValueError(f"Unsupported CI test: {ci_test_name}")

    ci_test_func = method_map[ci_test_name]
    mb = set(markov_blanket)

    for var in list(mb):
        Z = list(mb - {var})
        stat, p_val, dof = ci_test_func(X=var, Y=target, Z=Z, data=df, boolean=False)

        if verbose:
            logger.info(f"Tested: {var} ⊥ {target} | {Z} → p={p_val:.4f}, stat={stat:.2f}, dof={dof}")

        if p_val > alpha:
            mb.remove(var)
            if verbose:
                logger.info(f"Removed from MB: {var}")


    return mb


def IAMB_Algorithm(data: pd.DataFrame, 
                   target: str,
                   ci_test: str = "g_sq",
                   alpha: float = 0.05,
                   verbose: bool = False) -> set[str]:
    
    mb_growing = growing_phase(data, target=target, threshold=1e-4, verbose=verbose)
    mb_final = shrinking_phase(data, target=target, markov_blanket=mb_growing, ci_test_name=ci_test, alpha=alpha, verbose=verbose)

    if verbose:
        logger.info(f"Final Markov Blanket of '{target}': {mb_final}")
    
    return mb_final


if __name__ == "__main__":
    np.random.seed(42)
    df = pd.DataFrame({
        'A': [2, 1, 2, 2, 0],
        'B': [1, 1, 1, 1, 0],
        'C': [0, 0, 1, 1, 0],
        'D': [1, 0, 1, 2, 1],
        'E': [2, 0, 3, 4, 2],
        'F': [1, 2, 2, 2, 0],
        'G': [1, 1, 1, 1, 0],
        'H': [4, 5, 0, 3, 5]
    })

    result = IAMB_Algorithm(df, target='D', ci_test='chi_square', alpha=0.05)
    print("Markov Blanket of 'D':", result)
    
    
    
    
    
    
    
    
    
    
    