import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class EqualVolStrategy:
    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        # Adjust dollar weights to be proportional to 1/volatility
        dollar_weights = {
            ticker: 1 / np.sqrt(backtester.volatility.get(ticker, 1))
            for ticker in adj_universe
        }
        total_weight = sum(dollar_weights.values())
        # print(f'{total_weight=}')
        dollar_weights = {
            ticker: weight / total_weight for ticker, weight in dollar_weights.items()
        }
        return dollar_weights


class EqualDollarStrategy:
    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        return {ticker: 1 / len(adj_universe) for ticker in adj_universe}


class MinimumVarianceStrategy:
    def __init__(self):
        self._cov_ema = np.diag(np.full(1024, 0.02**2, dtype=np.float32))
        self.halflife = 30
        self.alpha = 1 - np.exp(np.log(0.5) / self.halflife)  # decay factor for EMA
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
                fix_i, fix_j = (
                    self._fix_id_by_ticker[tickers[i]],
                    self._fix_id_by_ticker[tickers[j]],
                )
                self._cov_ema[fix_i, fix_j] = (1 - self.alpha) * self._cov_ema[
                    fix_i, fix_j
                ] + self.alpha * cov_con
                self._cov_ema[fix_j, fix_i] = self._cov_ema[fix_i, fix_j]

    def objective_function(self, weights, cov_matrix):
        return weights @ cov_matrix @ weights

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

        num_assets = cov_matrix.shape[0]

        if num_assets == 0:
            return {ticker: 0 for ticker in adj_universe}

        # print(cov_matrix.shape)
        u, s, v = np.linalg.svd(cov_matrix)
        s[s < 1e-5] = 0
        cov_matrix = u @ np.diag(s) @ v
        # print(f'{u=}, {s=}, {v=}')
        # print(s[0], s[-1])

        initial_weights = [1.0 / num_assets] * num_assets
        bounds = [(0, 0.05) for _ in range(num_assets)]
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        result = minimize(
            self.objective_function,
            initial_weights,
            args=(cov_matrix,),
            bounds=bounds,
            constraints=constraints,
        )

        # Map optimal weights to tickers
        optimal_weights = {
            ticker: weight for ticker, weight in zip(adj_universe, result.x)
        }
        return optimal_weights


class MarkowitzStrategy:
    def __init__(self, risk_constant, return_estimate, vol_weighted, max_concentration):
        self._cov_ema = np.diag(np.full(1024, 0.02**2, dtype=np.float32))
        self.halflife = 30
        self.alpha = 1 - np.exp(np.log(0.5) / self.halflife)  # decay factor for EMA
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
                fix_i, fix_j = (
                    self._fix_id_by_ticker[tickers[i]],
                    self._fix_id_by_ticker[tickers[j]],
                )
                self._cov_ema[fix_i, fix_j] = (1 - self.alpha) * self._cov_ema[
                    fix_i, fix_j
                ] + self.alpha * cov_con
                self._cov_ema[fix_j, fix_i] = self._cov_ema[fix_i, fix_j]

    def objective_function(self, weights, return_vector, cov_matrix):
        return (
            weights @ return_vector
            - self._risk_constant * weights @ cov_matrix @ weights
        )

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

        num_assets = cov_matrix.shape[0]

        if num_assets == 0:
            return {ticker: 0 for ticker in adj_universe}

        return_vector = np.full(cov_matrix.shape[0], self.return_estimate)

        if self.vol_weighted:
            vols = np.diag(cov_matrix) ** 0.5
            return_vector *= vols / vols.mean()

        u, s, v = np.linalg.svd(cov_matrix)
        s[s < 1e-5] = 0
        cov_matrix = u @ np.diag(s) @ v
        initial_weights = [1.0 / num_assets] * num_assets
        bounds = [(0, self._max_concentration) for _ in range(num_assets)]
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        result = minimize(
            self.objective_function,
            initial_weights,
            args=(cov_matrix,),
            bounds=bounds,
            constraints=constraints,
        )

        # Map optimal weights to tickers
        optimal_weights = {
            ticker: weight for ticker, weight in zip(adj_universe, result.x)
        }
        return optimal_weights


class EqualVolContributionStrategy:
    def __init__(self):
        self._cov_ema = np.diag(np.full(1024, 0.02**2, dtype=np.float32))
        self.halflife = 30
        self.alpha = 1 - np.exp(np.log(0.5) / self.halflife)  # decay factor for EMA
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
                fix_i, fix_j = (
                    self._fix_id_by_ticker[tickers[i]],
                    self._fix_id_by_ticker[tickers[j]],
                )
                self._cov_ema[fix_i, fix_j] = (1 - self.alpha) * self._cov_ema[
                    fix_i, fix_j
                ] + self.alpha * cov_con
                self._cov_ema[fix_j, fix_i] = self._cov_ema[fix_i, fix_j]

    def volatility_contribution(self, weights, cov_matrix):
        # Calculate portfolio volatility
        port_variance = np.dot(weights, np.dot(cov_matrix, weights))
        # Marginal contribution of each asset to portfolio volatility
        marginal_contribution = np.dot(cov_matrix, weights)
        # Volatility contribution of each asset
        vol_contribution = weights * marginal_contribution
        return vol_contribution / np.sqrt(port_variance)

    def objective_function(self, weights, cov_matrix):
        contributions = self.volatility_contribution(weights, cov_matrix)
        avg_contribution = np.mean(contributions)
        # We want to minimize the sum of squared differences
        # between each asset's contribution and the average contribution
        return np.sum((contributions - avg_contribution) ** 2)

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

        num_assets = cov_matrix.shape[0]

        if num_assets == 0:
            return {ticker: 0 for ticker in adj_universe}

        # print(cov_matrix.shape)
        # u, s, v = np.linalg.svd(cov_matrix)
        # print(f'{u=}, {s=}, {v=}')

        initial_weights = [1.0 / num_assets] * num_assets
        bounds = [(0, 0.05) for _ in range(num_assets)]
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

        result = minimize(
            self.objective_function,
            initial_weights,
            args=(cov_matrix,),
            bounds=bounds,
            constraints=constraints,
        )

        # Map optimal weights to tickers
        optimal_weights = {
            ticker: weight for ticker, weight in zip(adj_universe, result.x)
        }
        return optimal_weights


class Backtester:
    def __init__(self, strategy, max_position_pct=0.05):
        self.strategy = strategy

        self.portfolio = {}
        self.portfolio_value_by_ticker = {}
        self.last_seen_by_ticker = {}
        self.n_day = 0
        self.portfolio_value = 0
        self.cash = 1000000
        self.date = datetime.date(1997, 6, 17)
        self.end_date = datetime.date(2009, 6, 17)
        self.daily_returns = []
        self.equity_curve = []
        self.universe = None
        self.snapshots = []
        self.halflife = 20
        self.alpha = 1 - np.exp(np.log(0.5) / self.halflife)  # decay factor for EMA
        self.prev_close = {}
        self.returns = {}  # stores the most recent return for each stock
        self.volatility = {}  # stores the EMA of volatility for each stock
        self.last_year = None
        self.max_position_pct = max_position_pct

        # SR comp
        self.alpha_yearly = 1 - np.exp(np.log(0.5) / 256)  # decay factor for EMA
        self.ema_ret = 0
        self.ema_vol = 0.02**2

        self.ret_all_time = 0
        self.vol_all_time = 0

    def _try_get_universe_by_date(self, date):
        filepath = os.path.join(
            "/Users/abdussamad/Documents/Github repos/portfolio-manager/RawDataStorage/UniversebyDate",
            f'universe{date.strftime("%Y-%m-%d")}.json',
        )
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r") as f:
            return json.load(f)

    def _try_get_prices_by_date(self, date):
        filepath = os.path.join(
            "/Users/abdussamad/Documents/Github repos/portfolio-manager/DataExtracts",
            f'daily_{date.strftime("%Y-%m-%d")}.json',
        )

        if not os.path.exists(filepath):
            return None

        res = []
        with open(filepath, "r") as f:
            for l in f:
                res.append(json.loads(l.strip()))
        return res

    @staticmethod
    def gini_coefficient(values):
        # Ensure values are sorted
        sorted_values = np.sort(values)
        n = len(sorted_values)

        # Compute the Lorenz curve's cumulative sum
        lorenz_curve = np.cumsum(sorted_values) / np.sum(sorted_values)

        # Compute the Gini coefficient
        gini = 1 - 2 * np.mean(lorenz_curve)
        return gini

    def _run_day(self):
        universe = self._try_get_universe_by_date(self.date)
        if universe:
            self.universe_date = self.date
            self.universe = universe

        prices = self._try_get_prices_by_date(self.date)
        if not prices:
            return

        for x in prices:
            self.last_seen_by_ticker[x["ticker"]] = self.n_day
        self.n_day += 1
        price_by_ticker = {x["ticker"]: x for x in prices}
        adj_universe = {
            x
            for x in self.universe
            if self.n_day - self.last_seen_by_ticker.get(x, 0) < 20
        }

        for ticker in adj_universe:
            if ticker not in price_by_ticker:
                continue
            if ticker not in self.volatility:
                self.volatility[ticker] = 0.02**2

        dollar_weights = self.strategy.get_dollar_weights(
            self, adj_universe, price_by_ticker
        )

        ## Enforce the maximum position constraint
        # total_weight = sum(dollar_weights.values())
        # for ticker, weight in dollar_weights.items():
        #    if weight / total_weight > self.max_position_pct:
        #        dollar_weights[ticker] = total_weight * self.max_position_pct
        ## Normalize weights after adjustments
        # total_weight = sum(dollar_weights.values())
        # dollar_weights = {ticker: weight/total_weight for ticker, weight in dollar_weights.items()}

        # Compute daily returns and update EMA of volatility
        for ticker in adj_universe:
            if ticker in price_by_ticker:
                pbt = price_by_ticker[ticker]
                adj_close = pbt["adjClose"]

                if ticker in self.prev_close:
                    daily_return = adj_close / self.prev_close[ticker] - 1
                    if ticker in self.volatility:
                        self.volatility[ticker] = (1 - self.alpha) * self.volatility[
                            ticker
                        ] + self.alpha * daily_return**2
                    self.returns[ticker] = daily_return
                self.prev_close[ticker] = adj_close

        portfolio_values_by_ticker = {}

        uni_plus_holdings = set(self.universe).union(
            set([k for k, v in self.portfolio.items() if v > 0])
        )
        for ticker in uni_plus_holdings:
            if ticker in price_by_ticker:
                pbt = price_by_ticker[ticker]
                adj_open = pbt["open"] * pbt["adjClose"] / pbt["close"]
                portfolio_values_by_ticker[ticker] = (
                    self.portfolio.get(ticker, 0) * adj_open
                )
            else:
                portfolio_values_by_ticker[ticker] = self.portfolio_value_by_ticker.get(
                    ticker, 0
                )

        portfolio_value = sum(portfolio_values_by_ticker.values()) + self.cash

        target_positions = {}
        for ticker in uni_plus_holdings:
            if ticker in price_by_ticker:
                if ticker in dollar_weights:
                    target_positions[ticker] = portfolio_value * dollar_weights[ticker]
                else:
                    target_positions[ticker] = 0

        turnover = 0

        if self.date.weekday() == 3:
            for ticker, tpos in target_positions.items():
                delta_usd = tpos - portfolio_values_by_ticker[ticker]
                pbt = price_by_ticker[ticker]
                fill_price = pbt["vwap"] * pbt["adjClose"] / pbt["close"]
                delta_shares = delta_usd / fill_price

                self.portfolio[ticker] = self.portfolio.get(ticker, 0) + delta_shares
                self.cash -= delta_usd
                turnover += abs(delta_usd)

        cost = turnover * 0.001

        for ticker in uni_plus_holdings:
            if ticker in price_by_ticker:
                pbt = price_by_ticker[ticker]
                portfolio_values_by_ticker[ticker] = (
                    self.portfolio.get(ticker, 0) * pbt["adjClose"]
                )

        portfolio_value = sum(portfolio_values_by_ticker.values()) + self.cash
        if self.snapshots:
            past_pv = self.snapshots[-1]["pv"]
            if past_pv:
                port_return = (portfolio_value - past_pv) / past_pv
                self.ema_ret = (
                    1 - self.alpha_yearly
                ) * self.ema_ret + self.alpha_yearly * port_return
                self.ema_vol = (
                    1 - self.alpha_yearly
                ) * self.ema_vol + self.alpha_yearly * port_return**2

        gini = self.gini_coefficient(list(dollar_weights.values()))

        snap = {
            "date": self.date.strftime("%Y-%m-%d"),
            "pv": portfolio_value,
            "cash": self.cash,
            "turnover": turnover,
            "cost": cost,
            "gini": gini,
            "n_stocks": len([x for x in self.portfolio.values() if x > 0]),
            "roll_ret": self.ema_ret * 256 * 100,
            "roll_sigma": self.ema_vol**0.5 * 16 * 100,
            "roll_sr": self.ema_ret / self.ema_vol**0.5 * 16,
            "portfolio": dict(self.portfolio),
        }
        self.snapshots.append(snap)
        # print(self.date, self.universe_date)
        # print([k for k in self.universe if k not in price_by_ticker])

        if ((self.last_year is not None) and (self.date.year != self.last_year)) or (
            self.date == self.end_date
        ):
            cagr, sharpe_ratio, volatility = self.compute_statistics(self.snapshots)
            print(self.date)
            print(f"CAGR: {cagr * 100:.2f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Volatility: {volatility * 100:.2f}%")
        self.last_year = self.date.year

    @staticmethod
    def compute_statistics(snapshots):
        # Extract daily portfolio values
        portfolio_values = [snap["pv"] for snap in snapshots if snap["pv"]]

        # Calculate daily returns based on portfolio values
        daily_returns = [
            (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            for i in range(1, len(portfolio_values))
        ]

        # 1. Calculate CAGR
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]

        def parse_date(date):
            return datetime.datetime.strptime(date, "%Y-%m-%d")

        num_years = (
            parse_date(snapshots[-1]["date"]) - parse_date(snapshots[0]["date"])
        ).days / 365.25
        cagr = (final_value / initial_value) ** (1 / num_years) - 1

        # 2. Calculate Sharpe Ratio
        average_return = np.mean(daily_returns) * 252
        volatility = np.std(daily_returns) * 252**0.5
        # Assuming risk-free rate to be 0
        risk_free_rate = 0
        sharpe_ratio = (average_return - risk_free_rate) / volatility

        return cagr, sharpe_ratio, volatility

    @staticmethod
    def compare_equity_curves(backtesters, labels):
        """
        Compare the equity curves of multiple backtest strategies on the same plot.

        Parameters:
        - backtesters (list): List of Backtester objects.
        - labels (list): Names/labels of the strategies for legend.
        """
        plt.figure(figsize=(12, 6))

        for bt, label in zip(backtesters, labels):
            dates = [
                datetime.datetime.strptime(snap["date"], "%Y-%m-%d")
                for snap in bt.snapshots
            ]
            portfolio_values = [snap["pv"] for snap in bt.snapshots]
            plt.plot(dates, portfolio_values, label=label, linewidth=2)

        plt.title("Equity Curve Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self, strategy):
        while self.date <= self.end_date:
            self._run_day()
            self.date += datetime.timedelta(days=1)

        with open(f"{strategy}.json", "w") as fo:
            for snap in self.snapshots:
                json.dump(snap, fo)
                fo.write("\n")

    def _get_price(self, ticker, date):
        try:
            return self.prices[ticker][date]
        except Exception:
            return None

    def _get_weights(self):
        return {ticker: 1 / len(self.universe) for ticker in self.universe}

    def _get_portfolio_value(self):
        return self.portfolio_value + self.cash

    def _get_portfolio_weights(self):
        return {
            ticker: self.portfolio[ticker] / self._get_portfolio_value()
            for ticker in self.portfolio
        }

    def _get_portfolio_returns(self):
        return self._get_portfolio_value() / self._get_portfolio_value() - 1

    def _get_cagr(self):
        return self._get_portfolio_value() / self.cash - 1


if __name__ == "__main__":
    # bt = Backtester(EqualDollarStrategy())
    # print(bt._try_get_universe_by_date(bt.date))
    # print(bt._try_get_universe_by_date(bt.date + datetime.timedelta(days=1)))
    # bt.run()

    # bt3 = Backtester(EqualVolContributionStrategy())
    # bt3.run("EqualVolContributionStrategy")

    bt1 = Backtester(EqualDollarStrategy())
    bt1.run("EqualDollarStrategy")

    bt2 = Backtester(EqualVolStrategy())
    bt2.run("EqualVolStrategy")

    bt4 = Backtester(MinimumVarianceStrategy())
    bt4.run("MinimumVarianceStrategy")

    bt5 = Backtesteer(MarkowitzStrategy())
    bt5.run("MarkowitzStrategy")

    Backtester.compare_equity_curves(
        [bt5, bt4, bt1, bt2],
        [
            "MarkowitzStrategy",
            "MinimumVarianceStrategy",
            "EqualDollarStrategy",
            "EqualVolStrategy",
        ],
    )
