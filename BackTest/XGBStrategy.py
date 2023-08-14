import numpy as np
import xgboost as xgb
import torch
import collections

class XGBStrategy:
    def __init__(self, retrain_every=252, regression=True):
        self._retrain_every = retrain_every
        self._trained = False
        self._regression = regression
        self._past_prices = collections.deque()
        self._past_preds = collections.deque()

        self._correct_preds = 0
        self._total_preds = 0
        self._total_positive_moves = 0

        self._ss_res = 0
        self._ss_targets = 0

        self.additional_information = {}

    #def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
    #    if (backtester.n_day + 1) % self._retrain_every == 0:
    #        self.train(backtester.price_history)
#
    #    if not self._trained:
    #        return {ticker: 0 for ticker in adj_universe}
    #    else:
    #        preds = self.infer(adj_universe, backtester.price_history)
    #        pos_weights = set(k for k, v in preds.items() if v > 0.01)
    #        print(len(adj_universe), len(pos_weights))
    #        if not pos_weights:
    #            return {ticker: 0 for ticker in adj_universe}
    #        return {ticker: 1/len(pos_weights) if ticker in pos_weights else 0 for ticker in adj_universe}

    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        if (backtester.n_day + 1) % self._retrain_every == 0:
            self.train(backtester.price_history)

        if not self._trained:
            return {ticker: 0 for ticker in adj_universe}

        preds = self.infer(adj_universe, backtester.price_history)
        self._past_preds.appendleft(dict(preds.items()))

        self._past_prices.appendleft(dict(price_by_ticker.items()))
        predicteds = []
        actuals = []

        if len(self._past_preds) > 5:
            for k, p in price_by_ticker.items():
                if k in self._past_preds[-1] and k in self._past_prices[-1]:
                    #print(self._past_prices[-1][k]['adjClose'], p['adjClose'])
                    actual_ret = (self._past_prices[-1][k]['adjClose'] - p['adjClose']) / self._past_prices[-1][k]['adjClose']
                    pred_ret = self._past_preds[-1][k]
                    predicteds.append(pred_ret)
                    actuals.append(actual_ret)

            self._past_preds.pop()
            self._past_prices.pop()

        actuals = np.array(actuals, dtype=np.float64)
        predicteds = np.array(predicteds, dtype=np.float64)

        if self._regression:
            pos_weights = {k: max(v, 0) for k, v in preds.items() if v > 0}  # using max to ensure no negative values
            total_weight = sum(pos_weights.values())
            if total_weight == 0:
                return {ticker: 0 for ticker in adj_universe}
            normalized_weights = {ticker: weight/total_weight for ticker, weight in pos_weights.items()}

            if actuals.size > 0:
                self._ss_res += ((actuals - predicteds)**2).sum()
                self._ss_targets += (actuals**2).sum()
                rsq = 1 - self._ss_res/self._ss_targets
                self.additional_information['rsq'] = rsq
        else:
            pred_values = list(preds.items())
            pred_values.sort(key=lambda x: x[1], reverse=True)
            weights = {k: 1 for k, v in pred_values[:len(pred_values)//10]}
            total_weight = sum(weights.values())
            normalized_weights = {ticker: weight/total_weight for ticker, weight in weights.items()}

            if actuals.size > 0:
                self._total_positive_moves += np.sum(actuals > 0)
                self._correct_preds += ((actuals > 0) == (predicteds > 0.5)).sum()
                self._total_preds += actuals.shape[0]
                self.additional_information['pos_ratio'] = self._total_positive_moves / self._total_preds
                self.additional_information['accuracy'] = self._correct_preds / self._total_preds

        return {ticker: normalized_weights.get(ticker, 0) for ticker in adj_universe}



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
        # print
        out = {t: y_pred[i] for i, t in enumerate(fix_id_by_adj_universe)}
        for t in adj_universe:
            if not t in out:
                out[t] = 0
        return out

    def train(self, price_history):
        price_matrix = price_history.get_price_matrix()

        idx_now = price_matrix.shape[0] - 1
        prediction_horizon = 5
        input_window = 15
        data_window = prediction_horizon + input_window + 1

        n_datapoints = 0

        for start_date_idx in range(0, idx_now - data_window):
            for stock_idx in range(0, price_matrix.shape[1]):
                if np.isnan(price_matrix[start_date_idx:start_date_idx+data_window, stock_idx]).any():
                    continue

                n_datapoints += 1

        X = np.zeros((n_datapoints, input_window))
        y = np.zeros((n_datapoints, 1))
        i = 0

        for start_date_idx in range(0, idx_now - data_window):
            for stock_idx in range(0, price_matrix.shape[1]):
                if np.isnan(price_matrix[start_date_idx:start_date_idx+data_window, stock_idx]).any():
                    continue

                price_series = price_matrix[start_date_idx:start_date_idx+data_window, stock_idx]
                X[i] = price_series[1:input_window+1]/price_series[0:input_window] - 1.0
                y[i] = price_series[-1]/price_series[input_window+1] - 1.0
                i += 1
        params = {}
        if torch.cuda.is_available():
            params['tree_method'] = 'gpu_hist'

        if not self._regression:
            y = (y > 0)
            params['objective'] = 'binary:logistic'

        dtrain = xgb.DMatrix(X, label=y)
        self._model = xgb.train(params, dtrain, num_boost_round=100)
        self._trained = True


