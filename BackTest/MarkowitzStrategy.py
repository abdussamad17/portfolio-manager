import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

def solve_qp(cov_matrix, return_vector=None, risk_constant=None, max_concentration=1, rec_counter=0):
    SCALE = 1000

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        with gp.Model(env=env) as m:
            w = m.addMVar(shape=cov_matrix.shape[0], lb=0, ub=max_concentration * SCALE, vtype=GRB.CONTINUOUS, name="w")
            if return_vector is not None:
                m.setObjective(w @ return_vector - risk_constant * w @ cov_matrix @ w, GRB.MAXIMIZE)
            else:
                m.setObjective(w @ cov_matrix @ w, GRB.MINIMIZE)
            m.addConstr(w.sum() == SCALE)
            try:
                m.optimize()
                # print(m.ObjVal)
            except gp.GurobiError as e:
                if rec_counter == 0:
                    name = f'numerical_{int(time.time())}_'
                    if return_vector is not None:
                        np.save(name + 'returns.npy', return_vector)
                    np.save(name + '_cov.npy', cov_matrix)
                if rec_counter == 3:
                    raise e
                print('Warn: Numerical issues, attempting to shrink cov matrix.')
                return solve_qp(cov_matrix + 5e-6 * np.eye(cov_matrix.shape[0]), return_vector, risk_constant, rec_counter=rec_counter+1)

            # Map optimal weights to tickers
            return w.X / SCALE

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
        weights = solve_qp(cov_matrix, return_vector, self._risk_constant, self._max_concentration)
        optimal_weights = {ticker: weight for ticker, weight in zip(adj_universe, weights)}
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
        weights = solve_qp(cov_matrix)
        optimal_weights = {ticker: weight for ticker, weight in zip(adj_universe, weights)}
        return optimal_weights