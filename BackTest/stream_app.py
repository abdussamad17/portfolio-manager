import pandas as pd
import datetime
import numpy as np
import json
import plotly.express as px
import streamlit as st
import s3fs
import plotly.graph_objs as go
import os
from dotenv import load_dotenv
import pickle
from Testback import *

load_dotenv()
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')


def download_file(bucket_name, file_path, local_file_path):
    fs = s3fs.S3FileSystem(anon=False)
    s3_file_path = f"{bucket_name}/{file_path}"
    fs.get(s3_file_path, local_file_path)
    return local_file_path


def plot_equity_curve(backtester, fig=None):
    """
    Plot the equity curve of a single backtest strategy using plotly.

    Parameters:
    - backtester (object): Backtester object.
    - fig: plotly.graph_objs.Figure
    """
    dates = [datetime.datetime.strptime(snap['date'], '%Y-%m-%d') for snap in backtester.snapshots]
    portfolio_values = [np.log10(snap['pv']) for snap in backtester.snapshots]

    if not fig:
        fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates, y=portfolio_values, mode='lines', name=backtester.strategy_name))

    fig.update_layout(
        legend=dict(
        x=0,
        y=0,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="white"
        ),
        ),
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        legend_title="Strategies",
        autosize=False,
        width=800,
        height=480,
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )
    return fig


st.title("Portfolio Backtest Visualizer")
st.sidebar.markdown("## Select Options Here")

strategies = ["EqualDollarStrategy", "EqualVolStrategy", "MinimumVarianceStrategy", "EqualVolContributionStrategy", "MarkowitzStrategy", "HRPStrategy", "XGBStrategy", "CNNStrategy"]

number_of_strategies = st.sidebar.slider("Number of strategies to compare", 1, len(strategies))

backtesters = []
for i in range(number_of_strategies):
    st.sidebar.markdown(f"## Strategy {i+1}")
    strategy_choice = st.sidebar.selectbox(f"Choose a Strategy {i+1}", strategies)

    full_strategy = f"{strategy_choice}"

    if strategy_choice == "MarkowitzStrategy":
        risk_constants = [0.5, 1, 2]
        risk_constant = st.sidebar.radio(f"Select Risk Constant for Strategy {i+1}", risk_constants)

        return_estimates = [0.000269, 0.00017, 0.00037]
        return_estimate = st.sidebar.radio(f"Select Return Estimate for Strategy {i+1}", return_estimates)

        vol_weighted = st.sidebar.checkbox(f"Volume Weighted for Strategy {i+1}?")
        max_concentrations = [0.05, 1]
        max_concentration = st.sidebar.radio(f"Select Max Concentration for Strategy {i+1}", max_concentrations)
        full_strategy += f",risk_constant={str(risk_constant)},return_estimate={str(return_estimate)},vol_weighted={str(vol_weighted)},max_concentration={str(max_concentration)}"

    elif strategy_choice == "HRPStrategy":
        linkage_methods = ["average", "single", "ward"]
        linkage_method = st.sidebar.selectbox(f"Select Linkage Method for Strategy {i+1}", linkage_methods)
        full_strategy += f",linkage_method={str(linkage_method)}"

    elif strategy_choice == "CNNStrategy":
        strategy_types = ["equalpositive", "equalpercent", "sigmoid"]
        strategy_type_choice = st.sidebar.selectbox(f"Select Strategy Type for Strategy {i+1}", strategy_types)
        full_strategy += f",strategy_type={str(strategy_type_choice)}"

    pickle_file_name = f"{full_strategy}.pkl"
    local_file_path = f"{pickle_file_name}"

    # Check if the file exists locally; if not, download from S3
    if not os.path.exists(local_file_path):
        bucket_name = "streamlitportfoliobucket"
        try:
            pickle_file = download_file(bucket_name, pickle_file_name, local_file_path)
        except FileNotFoundError:
            pickle_file = None

    # Check again if the file exists (either locally or just downloaded)
    if os.path.exists(local_file_path):
        with open(local_file_path, 'rb') as f:
            backtester = pickle.load(f)
            backtesters.append(backtester)
    else:
        st.info('This strategy is not yet available. Please try again later or choose another strategy', icon="🚨")

# Plot equity curve
fig = go.Figure()
for backtester in backtesters:
    fig = plot_equity_curve(backtester, fig)
st.plotly_chart(fig)
