import numpy as np
from scipy.optimize import minimize

def fit_mixture_model(X, y):
    # X: (n_sample, d)
    # y: (n_sample,)    

    def objective(theta):
        # theta: (d,)
        err = y - X @ theta # (*,)
        return np.sum(err**2) / max(n_sample, 1)

    def jacobian(theta):
        # theta: (d,)        
        jac = (2*(X.T@X)@theta - 2 * X.T@y)/max(n_sample, 1) # (d,)
        return jac

    def constraint(theta):
        return np.sum(theta) - 1.

    def constraint_jac(theta):
        return np.ones(theta.shape) # (d,)

    n_sample, d = X.shape
        
    bounds = [(0.,1.0,) for _ in range(d)]

    x0 = np.ones(d)/d

    problem = {'fun': objective, 'jac': jacobian, 'args': (), 'constraints': 
               {'type': 'eq', 'fun': constraint, 'jac': constraint_jac}, 
               'bounds': bounds}

    result = minimize(**problem, method='SLSQP', options={'disp': False}, x0=x0)

    return result.x, result