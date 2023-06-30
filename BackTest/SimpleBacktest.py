"""
Backtest flow:
- At day 0, start off with empty portfolio and some cash
- Generate weights, for 1/n weights that's literally 1/n for the size of the universe
- Compare the dollar value of the weights (proportional to your total portfolio value on the day) to the dollar value of the portfolio
- Trade the difference
- Outputs we care about is an equity curve (total portfolio value at a given time), CAGR (can be rolling, can be total), Sharpe ratio (ignore risk free rates for now, we might choose to simulate cash management as some of these portfolios will be investing in bonds)
"""
import os
import json
import datetime


class SimpleBacktest:

    def __init__(self):
        self.portfolio = {}
        self.portfolio_value_by_ticker = {}
        self.portfolio_value = 0
        self.cash = 1000000
        self.date = datetime.date(1997, 6, 17)
        self.end_date = datetime.date(1998, 6, 17)
        self.daily_returns = []
        self.equity_curve = []
        self.universe = None
        self.snapshots = []

    def _try_get_universe_by_date(self, date):
        filepath = os.path.join(
            '/Users/abdussamad/Documents/Github repos/portfolio-manager/RawDataStorage/UniversebyDate',
            f'universe{date.strftime("%Y-%m-%d")}.json'
        )
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r") as f:
            return json.load(f)

    def _try_get_prices_by_date(self, date):
        filepath = os.path.join(
            '/Users/abdussamad/Documents/Github repos/portfolio-manager/DataExtracts',
            f'daily_{date.strftime("%Y-%m-%d")}.json'
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
        price_by_ticker = {x['ticker']: x for x in prices}
        #print(f'{price_by_ticker=}')


        dollar_weights = {ticker: 1/len(self.universe) for ticker in self.universe}
        portfolio_values_by_ticker = {}

        uni_plus_holdings = set(self.universe).union(set([k for k, v in self.portfolio.items() if v > 0]))
        for ticker in uni_plus_holdings:
            if ticker in price_by_ticker:
                pbt = price_by_ticker[ticker]
                adj_open = pbt['open'] * pbt['adjClose'] / pbt['close']
                portfolio_values_by_ticker[ticker] = self.portfolio.get(ticker, 0) * adj_open
            else:
                portfolio_values_by_ticker[ticker] = self.portfolio_value_by_ticker.get(ticker, 0)

        #print(f'{portfolio_values_by_ticker=}')

        portfolio_value = sum(portfolio_values_by_ticker.values()) + self.cash
        #print(f'{portfolio_value=}')

        target_positions = {}
        for ticker in uni_plus_holdings:
            if ticker in price_by_ticker:
                if ticker in dollar_weights:
                    target_positions[ticker] = portfolio_value * dollar_weights[ticker]
                else:
                    target_positions[ticker] = 0
        #print(f'{target_positions=}')

        turnover = 0

        for ticker, tpos in target_positions.items():
            delta_usd = tpos - portfolio_values_by_ticker[ticker]
            pbt = price_by_ticker[ticker]
            fill_price = pbt['vwap'] * pbt['adjClose'] / pbt['close']
            delta_shares = delta_usd / fill_price

            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + delta_shares
            self.cash -= delta_usd
            turnover += abs(delta_shares)

        #print(f'{self.portfolio=}')

        cost = turnover * 0.001

        for ticker in uni_plus_holdings:
            if ticker in price_by_ticker:
                pbt = price_by_ticker[ticker]
                portfolio_values_by_ticker[ticker] = self.portfolio.get(ticker, 0) * pbt['adjClose']

        portfolio_value = sum(portfolio_values_by_ticker.values()) + self.cash
        snap = {
            'date': self.date.strftime('%Y-%m-%d'),
            'pv': portfolio_value,
            'cash': self.cash,
            'turnover': turnover,
            'cost': cost,
            'n_stocks': len([x for x in self.portfolio.values() if x > 0]),
            'portfolio': dict(self.portfolio),
        }
        self.snapshots.append(snap)
        #print(self.date, len(self.universe), len([k for k in self.universe if k in price_by_ticker]))
        print(self.date, self.universe_date)
        print([k for k in self.universe if k not in price_by_ticker])






    def run(self):
        while self.date <= self.end_date:
            self._run_day()
            self.date += datetime.timedelta(days=1)

        with open(f'backtest.json', "w") as fo:
            for snap in self.snapshots:
                json.dump(snap, fo)
                fo.write("\n")


    def _get_price(self, ticker, date):
        try:
            return self.prices[ticker][date]
        except Exception:
            return None

    def _get_weights(self):
        return {ticker: 1/len(self.universe) for ticker in self.universe}

    def _get_portfolio_value(self):
        return self.portfolio_value + self.cash

    def _get_portfolio_weights(self):
        return {ticker: self.portfolio[ticker]/self._get_portfolio_value() for ticker in self.portfolio}

    def _get_portfolio_returns(self):
        return self._get_portfolio_value()/self._get_portfolio_value() - 1

    def _get_cagr(self):
        return self._get_portfolio_value()/self.cash - 1

if __name__ == '__main__':
    bt = SimpleBacktest()
    #print(bt._try_get_universe_by_date(bt.date))
    #print(bt._try_get_universe_by_date(bt.date + datetime.timedelta(days=1)))
    bt.run()


