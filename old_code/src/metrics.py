import numpy as np

def entropy_from_logits(logits, base=2):
    """
    Entropía de la distribución softmax en cada paso (última dim = vocab).
    Retorna [batch, steps] en bits si base=2 (por defecto).
    """
    logits = logits - logits.max(axis=-1, keepdims=True)  # estabilidad numérica
    exp = np.exp(logits)
    p = exp / exp.sum(axis=-1, keepdims=True)
    logp = np.log(p + 1e-12)
    H = -(p * logp).sum(axis=-1)
    if base == 2:
        H = H / np.log(2.0)
    return H

def surprisal_of_true_token(logits, true_token_ids, base=2):
    """
    Surprisal = -log p(token verdadero) por paso.
    logits: [batch, steps, vocab]
    true_token_ids: [batch, steps] ints
    """
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    Z = exp.sum(axis=-1)
    idx = (*np.indices(true_token_ids.shape), true_token_ids)
    logp_true = logits[idx] - np.log(Z + 1e-12)
    S = -logp_true
    if base == 2:
        S = S / np.log(2.0)
    return S
