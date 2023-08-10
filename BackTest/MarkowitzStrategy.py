import numpy as np
import gurobipy as gp
from gurobipy import GRB

class MarkowitzStrategy:
    def __init__(self, risk_constant, return_estimate, vol_weighted, max_concentration):
        self._cov_ema = np.diag(np.full(1024, 0.02**2, dtype=np.float64))
        self.halflife = 30
        self.alpha = 1 - np.exp(np.log(0.5) / self.halflife) # decay factor for EMA
        self._fix_id_by_ticker = {}
        self._risk_constant = risk_constant
        self._return_estimate = return_estimate
        self._vol_weighted = vol_weighted
        self._max_concentration = max_concentration

    def compute_covariance_matrix(self, returns_by_ticker):
        for ticker in returns_by_ticker:
            if ticker not in self._fix_id_by_ticker:
                self._fix_id_by_ticker[ticker] = len(self._fix_id_by_ticker)

        tickers = list(returns_by_ticker.keys())
        n = len(tickers)
        for i in range(n):
            for j in range(i, n):
                cov_con = returns_by_ticker[tickers[i]] * returns_by_ticker[tickers[j]]
                fix_i, fix_j = self._fix_id_by_ticker[tickers[i]], self._fix_id_by_ticker[tickers[j]]
                self._cov_ema[fix_i, fix_j] = (1 - self.alpha) * self._cov_ema[fix_i, fix_j] + self.alpha * cov_con
                self._cov_ema[fix_j, fix_i] = self._cov_ema[fix_i, fix_j]

    # weights @ return_vector - self._risk_constant * weights @ cov_matrix @ weights

    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        returns_by_ticker = {ticker: backtester.returns[ticker] for ticker in adj_universe if ticker in backtester.returns}
        self.compute_covariance_matrix(returns_by_ticker)
        adj_estimated_universe = [x for x in adj_universe if x in self._fix_id_by_ticker]
        want_idx = [self._fix_id_by_ticker[x] for x in adj_estimated_universe]
        cov_matrix = self._cov_ema[want_idx][:, want_idx]

        num_assets = cov_matrix.shape[0]

        if num_assets == 0:
            return {ticker: 0 for ticker in adj_universe}

        return_vector = np.full(cov_matrix.shape[0], self._return_estimate)

        if self._vol_weighted:
            vols = np.diag(cov_matrix)**0.5
            return_vector *= vols/vols.mean()

        u, s, v = np.linalg.svd(cov_matrix)
        s[s < 1e-5] = 0
        cov_matrix = u @ np.diag(s) @ v

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()

            with gp.Model(env=env) as m:
                w = m.addMVar(shape=num_assets, lb=0, ub=self._max_concentration * 1000, vtype=GRB.CONTINUOUS, name="w")
                m.setObjective(w @ return_vector - self._risk_constant * w @ cov_matrix @ w, GRB.MAXIMIZE)
                m.addConstr(w.sum() == 1000)
                m.optimize()
                # print(m.ObjVal)
                # Map optimal weights to tickers
                optimal_weights = {ticker: weight/1000.0 for ticker, weight in zip(adj_universe, w.X)}
                return optimal_weights

class MinimumVarianceStrategy:
    def __init__(self):
        self._cov_ema = np.diag(np.full(1024, 0.02**2, dtype=np.float64))
        self.halflife = 30
        self.alpha = 1 - np.exp(np.log(0.5) / self.halflife) # decay factor for EMA
        self._fix_id_by_ticker = {}

    def compute_covariance_matrix(self, returns_by_ticker):
        for ticker in returns_by_ticker:
            if ticker not in self._fix_id_by_ticker:
                self._fix_id_by_ticker[ticker] = len(self._fix_id_by_ticker)

        tickers = list(returns_by_ticker.keys())
        n = len(tickers)
        for i in range(n):
            for j in range(i, n):
                cov_con = returns_by_ticker[tickers[i]] * returns_by_ticker[tickers[j]]
                fix_i, fix_j = self._fix_id_by_ticker[tickers[i]], self._fix_id_by_ticker[tickers[j]]
                self._cov_ema[fix_i, fix_j] = (1 - self.alpha) * self._cov_ema[fix_i, fix_j] + self.alpha * cov_con
                self._cov_ema[fix_j, fix_i] = self._cov_ema[fix_i, fix_j]

    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        returns_by_ticker = {ticker: backtester.returns[ticker] for ticker in adj_universe if ticker in backtester.returns}
        self.compute_covariance_matrix(returns_by_ticker)
        adj_estimated_universe = [x for x in adj_universe if x in self._fix_id_by_ticker]
        want_idx = [self._fix_id_by_ticker[x] for x in adj_estimated_universe]
        cov_matrix = self._cov_ema[want_idx][:, want_idx]

        num_assets = cov_matrix.shape[0]

        if num_assets == 0:
            return {ticker: 0 for ticker in adj_universe}

        u, s, v = np.linalg.svd(cov_matrix)
        s[s < 1e-5] = 0
        cov_matrix = u @ np.diag(s) @ v

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()

            with gp.Model(env=env) as m:
                w = m.addMVar(shape=num_assets, lb=0, vtype=GRB.CONTINUOUS, name="w")
                m.setObjective(w @ cov_matrix @ w, GRB.MINIMIZE)
                m.addConstr(w.sum() == 1000)
                m.optimize()
                # print(m.ObjVal)
                # Map optimal weights to tickers
                optimal_weights = {ticker: weight/1000.0 for ticker, weight in zip(adj_universe, w.X)}
                return optimal_weights