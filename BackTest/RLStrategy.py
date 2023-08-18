import numpy as np
import torch


class RLModel:
    def __init__(self):
        self._value_network = None
        self._actor_network = None

    def _get_backbone(self):
        return [
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20),
        ]

    def train(self, price_history):
        self._value_network = torch.nn.Sequential(self._get_backbone())
        self._actor_network = torch.nn.Sequential(
            self._get_backbone() + [torch.nn.Sigmoid()]
        )

    def outputs(self, price_history, adj_universe):
        return {t: 0 for t in adj_universe}


class RLStrategy:
    def __init__(self, retrain_every=252):
        self._retrain_every = retrain_every
        self._trained = False

    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        if (backtester.n_day + 1) % self.retrain_every == 0:
            self.train(backtester.price_history)

        if not self._trained:
            return {ticker: 0 for ticker in adj_universe}
        else:
            return self._model.outputs(backtester.price_history, adj_universe)

    def train(self, price_history):
        return price_history
