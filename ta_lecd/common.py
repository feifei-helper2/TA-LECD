import numpy as np

def relabel0(labels):
    _, inverse = np.unique(np.asarray(labels).ravel(), return_inverse=True)
    return inverse.astype(np.int32)
