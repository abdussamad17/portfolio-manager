import os
import time
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import s3fs
from dotenv import load_dotenv
from Testback import *

GRAPH_WIDTH = 1000
GRAPH_HEIGHT = 600

load_dotenv()
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

model_results_path = "model_results"


def download_file(bucket_name, file_path, local_file_path):
    fs = s3fs.S3FileSystem(anon=False)
    s3_file_path = f"{bucket_name}/{file_path}"
    fs.get(s3_file_path, local_file_path)
    return local_file_path


def load_json_data(file_path):
    local_path = os.path.join(model_results_path, file_path)
    if not os.path.exists(local_path):
        bucket_name = "streamlitportfoliobucket"
        download_file(bucket_name, file_path, local_path)

    if os.path.exists(local_path):
        with open(local_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
        return data
    else:
        st.info('This strategy is not yet available. Please try again later or choose another strategy', icon="🚨")


def setup_strategy(strategy_choice, i):
    full_strategy = f"{strategy_choice}"
    if strategy_choice == "MarkowitzStrategy":
        # Setup specific parameters for MarkowitzStrategy
        risk_constants = [0.5, 1, 2]
        risk_constant = st.sidebar.radio(f"Select Risk Constant for Strategy {i+1}", risk_constants)
        return_estimates = [0.000269, 0.00017, 0.00037]
        return_estimate = st.sidebar.radio(f"Select Return Estimate for Strategy {i+1}", return_estimates)
        vol_weighted = st.sidebar.checkbox(f"Volume Weighted for Strategy {i+1}?")
        max_concentrations = [0.05, 1]
        max_concentration = st.sidebar.radio(f"Select Max Concentration for Strategy {i+1}", max_concentrations)
        full_strategy += f",risk_constant={str(risk_constant)},return_estimate={str(return_estimate)},vol_weighted={str(vol_weighted)},max_concentration={str(max_concentration)}"
    elif strategy_choice == "HRPStrategy":
        # Setup specific parameters for HRPStrategy
        linkage_methods = ["average", "single", "ward"]
        linkage_method = st.sidebar.selectbox(f"Select Linkage Method for Strategy {i+1}", linkage_methods)
        full_strategy += f",linkage_method={str(linkage_method)}"
    elif strategy_choice == "CNNStrategy":
        # Setup specific parameters for CNNStrategy
        strategy_types = ["marketindicator", "equalpercent", "sigmoid"]
        strategy_type_choice = st.sidebar.selectbox(f"Select Strategy Type for Strategy {i+1}", strategy_types)
        full_strategy += f",strategy_type={str(strategy_type_choice)}"
    elif strategy_choice == "XGBStrategy":
        # Setup specific parameters for XGBStrategy
        regression = st.sidebar.checkbox(f"Regression for strat {i+1}?")
        full_strategy += f",regression={str(regression)}"
    return full_strategy


def display_json_data(json_data_list, backtesters):
    metrics_fig = go.Figure()
    for json_data, backtester in zip(json_data_list, backtesters):
        df = pd.DataFrame(json_data, columns=['date', 'roll_sigma', 'cash', 'roll_sr'])
        df['date'] = pd.to_datetime(df['date'])
        metrics_fig.add_trace(go.Scatter(x=df['date'], y=df['roll_sr'], mode='lines', name=f'{backtester.strategy_name} - Rolling Sharpe'))
    metrics_fig.update_layout(title="Portfolio Metrics Over Time", xaxis_title="Date", width=GRAPH_WIDTH, height=GRAPH_HEIGHT,
                              legend=dict(x=0, y=0, traceorder="normal", font=dict(family="sans-serif", size=12, color="white")))
    st.plotly_chart(metrics_fig)

    selected_date = st.date_input("Choose a date to view portfolio constituents", value=pd.to_datetime(json_data_list[0][0]['date']),
                                  min_value=pd.to_datetime(json_data_list[0][0]['date']), max_value=pd.to_datetime(json_data_list[0][-1]['date']))
    constituents_fig = go.Figure()
    scaling_factor = 10
    for json_data, backtester in zip(json_data_list, backtesters):
        selected_data = [item for item in json_data if item["date"] == selected_date.strftime('%Y-%m-%d')][0]
        portfolio = selected_data['portfolio']
        portfolio_df = pd.DataFrame(list(portfolio.items()), columns=['Asset', 'Value'])
        portfolio_df['Weight'] = (portfolio_df['Value'] / portfolio_df['Value'].sum() * 100).astype(float) * scaling_factor
        portfolio_df['Strategy'] = backtester.strategy_name
        constituents_fig.add_trace(go.Scatter(x=portfolio_df['Asset'], y=portfolio_df['Value'], mode='markers', marker=dict(size=portfolio_df['Weight'], sizemode='diameter'),
                                              text=portfolio_df['Asset'], name=backtester.strategy_name))
    constituents_fig.update_layout(title="Portfolio Constituents", xaxis_title="Asset", yaxis_title="Value", showlegend=True, width=1000, height=600)
    constituents_fig.update_traces(textposition='top center')
    st.plotly_chart(constituents_fig)


def plot_equity_curve(backtester, fig=None):
    dates = [datetime.datetime.strptime(snap['date'], '%Y-%m-%d') for snap in backtester.snapshots]
    portfolio_values = [np.log10(snap['pv']) for snap in backtester.snapshots]
    if not fig:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=portfolio_values, mode='lines', name=backtester.strategy_name))
    fig.update_layout(
        legend=dict(x=0, y=0, traceorder="normal", font=dict(family="sans-serif", size=12, color="white")),
        title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value", legend_title="Strategies", autosize=False,
        width=GRAPH_WIDTH, height=GRAPH_HEIGHT, margin=dict(l=50, r=50, b=100, t=100, pad=4))
    return fig


def plot_comparisons(json_data_list, backtesters):
    rolling_metrics_fig = go.Figure()
    universe_size_fig = go.Figure()
    gini_fig = go.Figure()
    for json_data, backtester in zip(json_data_list, backtesters):
        df = pd.DataFrame(json_data)
        df['date'] = pd.to_datetime(df['date'])
        rolling_metrics_fig.add_trace(go.Scatter(x=df['date'], y=df['roll_sigma'], mode='lines', name=f'{backtester.strategy_name} - Rolling Sigma'))
        rolling_metrics_fig.add_trace(go.Scatter(x=df['date'], y=df['roll_sr'], mode='lines', name=f'{backtester.strategy_name} - Rolling Sharpe Ratio'))
        universe_size_fig.add_trace(go.Scatter(x=df['date'], y=df['n_stocks'], mode='lines', name=f'{backtester.strategy_name} - n_stocks'))
        universe_size_fig.add_trace(go.Scatter(x=df['date'], y=df['adj_uni_size'], mode='lines', name=f'{backtester.strategy_name} - Adjusted Universe Size', line=dict(dash='dash')))
        gini_fig.add_trace(go.Scatter(x=df['date'], y=df['gini'], mode='lines', name=f'{backtester.strategy_name}'))

    rolling_metrics_fig.update_layout(title="Rolling Metrics Over Time: Comparison", xaxis_title="Date")
    st.plotly_chart(rolling_metrics_fig)

    universe_size_fig.update_layout(title="Number of Stocks vs. Adjusted Universe Size: Comparison", xaxis_title="Date", yaxis_title="Size")
    st.plotly_chart(universe_size_fig)

    gini_fig.update_layout(title="Gini Coefficient Over Time: Comparison", xaxis_title="Date", yaxis_title="Gini Coefficient")
    st.plotly_chart(gini_fig)


# Main code logic
st.title("Portfolio Backtest Visualizer")
strategies = ["EqualDollarStrategy", "EqualVolStrategy", "MinimumVarianceStrategy", "EqualVolContributionStrategy", "MarkowitzStrategy", "HRPStrategy", "XGBStrategy", "CNNStrategy"]
number_of_strategies = st.sidebar.slider("Number of strategies to compare", 1, len(strategies))
json_data_list = []
backtesters = []

for i in range(number_of_strategies):
    strategy_choice = st.sidebar.selectbox(f"Choose a Strategy {i+1}", strategies)
    full_strategy = setup_strategy(strategy_choice, i)
    pickle_file_name = f"{full_strategy}.pkl"
    local_file_path = os.path.join(model_results_path, f"{pickle_file_name}")
    json_file_name = f"{full_strategy}.json"
    json_data = load_json_data(json_file_name)
    json_data_list.append(json_data)
    if not os.path.exists(local_file_path):
        bucket_name = "streamlitportfoliobucket"
        download_file(bucket_name, pickle_file_name, local_file_path)
    if os.path.exists(local_file_path):
        with open(local_file_path, 'rb') as f:
            backtester = pickle.load(f)
            backtesters.append(backtester)

tabs = ["Equity Curves", "Portfolio Metrics", "Comparisons"]
selected_tab = st.tabs(tabs)

if selected_tab == "Equity Curve":
    fig = go.Figure()
    for backtester in backtesters:
        fig = plot_equity_curve(backtester, fig)
    st.plotly_chart(fig)

elif selected_tab == "Portfolio Metrics":
    display_json_data(json_data_list, backtesters)

elif selected_tab == "Comparisons":
    plot_comparisons(json_data_list, backtesters)
