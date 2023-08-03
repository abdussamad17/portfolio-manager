import numpy as np
import torch

class CNNStrategy:
    def __init__(self, retrain_every=252):
        self._retrain_every = retrain_every
        self._trained = False

    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        if (backtester.n_day + 1) % self._retrain_every == 0:
            self.train(backtester.price_history)

        if not self._trained:
            return {ticker: 0 for ticker in adj_universe}
        else:
            preds = self.infer(adj_universe, backtester.price_history)
            pos_weights = set(k for k, v in preds.items() if v > 0.01)
            print(len(adj_universe), len(pos_weights))
            if not pos_weights:
                return {ticker: 0 for ticker in adj_universe}
            return {ticker: 1/len(pos_weights) if ticker in pos_weights else 0 for ticker in adj_universe}

    def infer(self, adj_universe, price_history):
        price_matrix = price_history.get_price_matrix()

        idx_now = price_matrix.shape[0] - 1
        prediction_horizon = 5
        input_window = 15 + 1
        data_window = prediction_horizon + input_window + 1

        fix_id_by_adj_universe = {t: price_history._fix_id_by_ticker[t] for t in adj_universe if t in price_history._fix_id_by_ticker}
        fix_id_by_adj_universe = {k: v for k, v in fix_id_by_adj_universe.items() if not
            np.isnan(price_matrix[-input_window:, v]).any()}
        X_pred = np.zeros((len(fix_id_by_adj_universe), input_window - 1))

        for i, (t, v) in enumerate(fix_id_by_adj_universe.items()):
            price_series = price_matrix[-input_window:, v]
            X_pred[i, :] = price_series[1:input_window]/price_series[0:input_window-1] - 1.0

        y_pred = self._model.predict(xgb.DMatrix(X_pred))

        # print(y_pred)
        out = {t: y_pred[i] for i, t in enumerate(fix_id_by_adj_universe)}
        for t in adj_universe:
            if not t in out:
                out[t] = 0
        return out

    def train(self, price_history):
        input_types =  [
            'adj_close',
            'adj_open',
            'adj_high',
            'adj_low',
            'vol',
        ]
        price_matrix_by_type = {k: price_history.get_price_matrix(k) for k in input_types}
        price_matrix = price_matrix_by_type['adj_close']

        idx_now = price_matrix.shape[0] - 1
        prediction_horizon = 5
        input_window = 15
        data_window = prediction_horizon + input_window + 1
        ema_const = 1 - np.exp(np.log(0.5) / 7.5) # decay factor for EMA

        n_datapoints = 0

        for start_date_idx in range(0, idx_now - data_window):
            for stock_idx in range(0, price_matrix.shape[1]):
                if np.isnan(price_matrix[start_date_idx:start_date_idx+data_window, stock_idx]).any():
                    continue

                n_datapoints += 1

        X = np.zeros((n_datapoints, input_window, len(input_types) + 1))
        ema_idx = len(input_types)
        y = np.zeros((n_datapoints, 1))
        i = 0

        for stock_idx in range(0, price_matrix.shape[1]):
            adj_close_ema = np.nan

            for start_date_idx in range(0, idx_now - data_window):
                if not np.isnan(price_matrix[start_date_idx, stock_idx]):
                    if np.isnan(adj_close_ema):
                        adj_close_ema = price_matrix[start_date_idx, stock_idx]
                    else:
                        adj_close_ema = (1 - ema_const) * adj_close_ema + ema_const * price_matrix[start_date_idx, stock_idx]

                if np.isnan(price_matrix[start_date_idx:start_date_idx+data_window, stock_idx]).any():
                    continue

                for d, t in enumerate(input_types):
                    submatrix = price_matrix_by_type[t]
                    price_series = submatrix[start_date_idx:start_date_idx+data_window, stock_idx]
                    X[i, :, d] = price_series[:input_window]
                X[i, 0, ema_idx] = adj_close_ema
                for t in range(1, X.shape[1]):
                    X[i, t, ema_idx] = (1 - ema_const) * X[i, t-1, ema_idx] + ema_const * X[i, t, 0]

                y[i] = price_series[-1]/price_series[input_window+1] - 1.0
                i += 1

        np.savez('inputs.npz', X, y)

        # dtrain = xgb.DMatrix(X, label=y)
        # self._model = xgb.train({}, dtrain, num_boost_round=100)
        # self._trained = True


