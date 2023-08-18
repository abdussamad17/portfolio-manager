import os
import json
import datetime
import numpy as np


class VolTargetBacktest:
    def __init__(self):
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
        self.returns = {}
        self.volatility = {}  # EMA of volatility for each stock

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
                    else:
                        self.volatility[ticker] = 0.02**2
                    self.returns[ticker] = daily_return
                self.prev_close[ticker] = adj_close

        # Adjust dollar weights to be proportional to 1/volatility
        dollar_weights = {
            ticker: 1 / np.sqrt(self.volatility.get(ticker, 1))
            for ticker in adj_universe
        }
        total_weight = sum(dollar_weights.values())
        print(f"{total_weight=}")
        dollar_weights = {
            ticker: weight / total_weight for ticker, weight in dollar_weights.items()
        }

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

        snap = {
            "date": self.date.strftime("%Y-%m-%d"),
            "pv": portfolio_value,
            "cash": self.cash,
            "turnover": turnover,
            "cost": cost,
            "n_stocks": len([x for x in self.portfolio.values() if x > 0]),
            "roll_ret": self.ema_ret * 256 * 100,
            "roll_sigma": self.ema_vol**0.5 * 16 * 100,
            "roll_sr": self.ema_ret / self.ema_vol**0.5 * 16,
            "portfolio": dict(self.portfolio),
        }
        self.snapshots.append(snap)
        # print(self.date, self.universe_date)
        # print([k for k in self.universe if k not in price_by_ticker])

    def run(self):
        while self.date <= self.end_date:
            self._run_day()
            self.date += datetime.timedelta(days=1)

        with open(f"VolatilityBacktest.json", "w") as fo:
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
    bt = VolTargetBacktest()
    # print(bt._try_get_universe_by_date(bt.date))
    # print(bt._try_get_universe_by_date(bt.date + datetime.timedelta(days=1)))
    bt.run()
