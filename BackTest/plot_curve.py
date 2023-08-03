import matplotlib.pyplot as plt
import pickle
import sys

from Testback import *

def compare_equity_curves(backtesters):
    """
    Compare the equity curves of multiple backtest strategies on the same plot.

    Parameters:
    - backtesters (list): List of Backtester objects.
    """
    labels = [b.strategy_name for b in backtesters]
    plt.figure(figsize=(12, 6))

    for bt, label in zip(backtesters, labels):
        dates = [datetime.datetime.strptime(snap['date'], '%Y-%m-%d') for snap in bt.snapshots]
        portfolio_values = [np.log10(snap['pv']) for snap in bt.snapshots]
        plt.plot(dates, portfolio_values, label=label, linewidth=2)

    plt.title("Equity Curve Comparison")
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pickles = sys.argv[1:]
    backtesters = []

    for pickle_file in pickles:
        with open(pickle_file, 'rb') as f:
            backtester = pickle.load(f)
            backtesters.append(backtester)
    compare_equity_curves(backtesters)