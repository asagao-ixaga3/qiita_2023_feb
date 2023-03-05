import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeCV
from utils import fit_mixture_model

class UclkAgent(BaseEstimator):

    def __init__(self, nx, na, scale = 0.5, d = 2**3) -> None:
        super().__init__()

        # feature map parameters
        self.na = na
        self.scale = scale
        self.d = d
        self.centers = 6*np.random.rand(d,nx)-3 # (d,nx)

        self.V = np.zeros(d+1) # (d+1,)
        self.Q = np.zeros((d+1,na)) # (d+1,na)
        self.Theta = np.ones((d,na))/d # (d,na)

    def feat_map(self, X):
        # (..., nx)
        nx = X.shape[-1]
        return multivariate_normal(mean=np.zeros(nx),cov=self.scale**2).pdf(
                X[...,None,:]-self.centers) # (...,d)

    def inner_product(self, X):
        # X: (..., nx)
        nx = X.shape[-1]
        tmp = X[...,None,None,:] + self.centers[:,None,:] \
            - self.centers[None,:,:] # (...,d,d,nx)
        psi_main = multivariate_normal(mean=np.zeros(nx),
            cov=2*self.scale**2).pdf(tmp) # (...,d,d)
        psi_bias = np.ones(shape=(*psi_main.shape[:-1],1)) # (...,d,1)
        return np.concatenate((psi_main, psi_bias), axis=-1) # (...,d,d+1)

    def predict_q(self, S):
        Phi = self.feat_map(S) # (N,d)
        return Phi@self.Q[:-1,:] + self.Q[-1,:] # (N,na)

    def predict_v(self, S):
        Phi = self.feat_map(S) # (N,d)
        return Phi@self.V[:-1] + self.V[-1] # (N,)

    def predict_next_state(self, S, A):
        # S: (..., nx), A: (...)

        # self.Theta # (d,na)
        # self.centers # (d,nx)
        return S + self.Theta.T[A,:] @ self.centers 
        # (...,d) @ (d,nx) -> (...,nx)

    def fit(self,S0, A0, S1, S_tld, R_tld, n_rounds=2, gamma = 0.9, eps=1e-8, 
        n_itr = 2**3):
        # S0,S1: (N, nx), A0: (N,), S_tld: (N_tld,nx), R_tld: (N_tld, na)
        
        d = self.d
        na = self.na

        N, nx = S0.shape
        N_tld = S_tld.shape[0]
        assert S1.shape == (N, nx)
        assert S_tld.shape == (N_tld, nx)
        assert A0.shape == (N,)

        V = np.zeros(d+1) # (d+1,)
        Q = np.zeros((d+1,na)) # (d+1,na)
        Theta = np.ones((d,na))/d # (d,na)

        Phi1 = self.feat_map(S1) # (N,d)
        Phi_tld = self.feat_map(S_tld) # (N_tld,d)

        Psi0 = self.inner_product(S0) # (N,d,d+1)
        Psi_tld = self.inner_product(S_tld) # (N_tld,d,d+1)
        
        for _ in range(n_itr):
            # run EVI process            
            for _ in range(n_rounds):
                Y = R_tld + gamma * (Psi_tld @ V) @ Theta # (N~,na)     
                model = RidgeCV().fit(Phi_tld, Y)
                Q[:-1,:] = model.coef_.T # (d, na)
                Q[-1,:] = model.intercept_ # (na,)
                Y = np.max(model.predict(Phi_tld), axis=-1) # (N~,)
                model = RidgeCV().fit(Phi_tld, Y)
                V[:-1] = model.coef_ # (d,)
                V[-1] = model.intercept_ # (,)

            # Solve theta
            Y = np.concatenate((Phi1, np.ones((N,1))),axis=-1) @ V # (N,)
            X = Psi0 @ V # (N,d)
            for a in range(na):                
                Ya = Y[A0==a] # (*,)
                Xa = X[A0==a,:] # (*,d)
                theta, res = fit_mixture_model(Xa, Ya)
                assert res.success, res.message
                Theta[:,a] = theta[:]

        self.Theta[:,:] = Theta[:,:] # (d,na)
        self.V[:] = V[:] # (d+1,)
        self.Q[:,:] = Q[:,:] # (d,na)