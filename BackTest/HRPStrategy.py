import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


class HRPStrategy:
    def __init__(self, linkage_method):
        self._cov_ema = np.diag(np.full(1024, 0.02**2, dtype=np.float32))
        self.halflife = 30
        self.alpha = 1 - np.exp(np.log(0.5) / self.halflife)  # decay factor for EMA
        self._fix_id_by_ticker = {}
        self._linkage_method = linkage_method

    def compute_covariance_matrix(self, returns_by_ticker):
        for ticker in returns_by_ticker:
            if ticker not in self._fix_id_by_ticker:
                self._fix_id_by_ticker[ticker] = len(self._fix_id_by_ticker)

        tickers = list(returns_by_ticker.keys())
        n = len(tickers)
        for i in range(n):
            for j in range(i, n):
                cov_con = returns_by_ticker[tickers[i]] * returns_by_ticker[tickers[j]]
                fix_i, fix_j = (
                    self._fix_id_by_ticker[tickers[i]],
                    self._fix_id_by_ticker[tickers[j]],
                )
                self._cov_ema[fix_i, fix_j] = (1 - self.alpha) * self._cov_ema[
                    fix_i, fix_j
                ] + self.alpha * cov_con
                self._cov_ema[fix_j, fix_i] = self._cov_ema[fix_i, fix_j]

    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        returns_by_ticker = {
            ticker: backtester.returns[ticker]
            for ticker in adj_universe
            if ticker in backtester.returns
        }
        self.compute_covariance_matrix(returns_by_ticker)
        adj_estimated_universe = [
            x for x in adj_universe if x in self._fix_id_by_ticker
        ]
        want_idx = [self._fix_id_by_ticker[x] for x in adj_estimated_universe]
        cov_matrix = self._cov_ema[want_idx][:, want_idx]
        ticker_by_cov_id = {i: n for i, n in enumerate(adj_estimated_universe)}

        num_assets = cov_matrix.shape[0]

        if num_assets < 10:
            return {ticker: 0 for ticker in adj_universe}

        vars = np.diag(cov_matrix).reshape(-1, 1)
        corr_matrix = cov_matrix / (vars @ vars.T) ** 0.5
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        ordered_dist_mat, res_order, res_linkage = self.compute_serial_matrix(
            dist_matrix, method=self._linkage_method
        )
        hrp_weights = self.compute_HRP_weights(cov_matrix, res_order)

        return {ticker_by_cov_id[idx]: w for idx, w in hrp_weights.items()}

    # Extracted from https://gmarti.gitlab.io/qfin/2018/10/02/hierarchical-risk-parity-part-1.html
    @staticmethod
    def seriation(Z, N, cur_index):
        if cur_index < N:
            return [cur_index]
        else:
            left = int(Z[cur_index - N, 0])
            right = int(Z[cur_index - N, 1])
            return HRPStrategy.seriation(Z, N, left) + HRPStrategy.seriation(
                Z, N, right
            )

    # Extracted from https://gmarti.gitlab.io/qfin/2018/10/02/hierarchical-risk-parity-part-1.html
    @staticmethod
    def compute_serial_matrix(dist_mat, method="ward"):
        N = len(dist_mat)
        flat_dist_mat = squareform(dist_mat)
        res_linkage = linkage(flat_dist_mat, method=method)
        res_order = HRPStrategy.seriation(res_linkage, N, N + N - 2)
        seriated_dist = np.zeros((N, N))
        a, b = np.triu_indices(N, k=1)
        seriated_dist[a, b] = dist_mat[
            [res_order[i] for i in a], [res_order[j] for j in b]
        ]
        seriated_dist[b, a] = seriated_dist[a, b]

        return seriated_dist, res_order, res_linkage

    @staticmethod
    def compute_HRP_weights(covariances, res_order):
        weights = pd.Series(1, index=res_order)
        clustered_alphas = [res_order]

        vars = np.diag(covariances)

        while len(clustered_alphas) > 0:
            clustered_alphas = [
                cluster[start:end]
                for cluster in clustered_alphas
                for start, end in (
                    (0, len(cluster) // 2),
                    (len(cluster) // 2, len(cluster)),
                )
                if len(cluster) > 1
            ]
            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]

                left_subcovar = covariances[left_cluster][:, left_cluster]
                inv_diag = 1 / np.diag(left_subcovar)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

                right_subcovar = covariances[right_cluster][:, right_cluster]
                inv_diag = 1 / np.diag(right_subcovar)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

                alloc_factor = 1 - left_cluster_var / (
                    left_cluster_var + right_cluster_var
                )

                weights[left_cluster] *= alloc_factor
                weights[right_cluster] *= 1 - alloc_factor

        return weights
