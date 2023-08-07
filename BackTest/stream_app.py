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

def load_json_data(file_path):
    if not os.path.exists(file_path):
        bucket_name = "streamlitportfoliobucket"
        try:
            json_file = download_file(bucket_name, file_path, file_path)
        except FileNotFoundError:
            json_file = None

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
        return data
    else:
        st.info('This strategy is not yet available. Please try again later or choose another strategy', icon="🚨")

# Function to display the JSON data in Streamlit
def display_json_data(json_data_list, backtesters):
    # Initialize figure for Portfolio Metrics
    metrics_fig = go.Figure()

    # Loop through each strategy's json_data and backtester
    for json_data, backtester in zip(json_data_list, backtesters):
        # Convert json_data to DataFrame
        df = pd.DataFrame(json_data)
        df['date'] = pd.to_datetime(df['date'])

        # Plotting Portfolio Metrics for each strategy
        metrics_fig.add_trace(go.Scatter(x=df['date'], y=df['pv'], mode='lines', name=f'{backtester.strategy_name} - Portfolio Value'))
        metrics_fig.add_trace(go.Scatter(x=df['date'], y=df['cash'], mode='lines', name=f'{backtester.strategy_name} - Cash'))
        # Add more metrics if needed

    # Update layout and plot the combined metrics figure
    metrics_fig.update_layout(title="Portfolio Metrics Over Time", xaxis_title="Date")
    st.plotly_chart(metrics_fig)

    # Ask the user to choose a date for the portfolio constituents
    selected_date = st.date_input("Choose a date to view portfolio constituents", value=pd.to_datetime(json_data_list[0][0]['date']), min_value=pd.to_datetime(json_data_list[0][0]['date']), max_value=pd.to_datetime(json_data_list[0][-1]['date']))

    # Initialize figure for Portfolio Constituents
    constituents_fig = go.Figure()

    scaling_factor = 10
    plot_width = 1000
    plot_height = 600



    # Loop through each strategy's json_data to plot Portfolio Constituents
    for json_data, backtester in zip(json_data_list, backtesters):
        selected_data = [item for item in json_data if item["date"] == selected_date.strftime('%Y-%m-%d')][0]
        portfolio = selected_data['portfolio']
        portfolio_df = pd.DataFrame(list(portfolio.items()), columns=['Asset', 'Value'])
        portfolio_df['Weight'] = (portfolio_df['Value'] / portfolio_df['Value'].sum() * 100).astype(float) * scaling_factor
        portfolio_df['Strategy'] = backtester.strategy_name

        # Adding scatter trace for portfolio constituents
        constituents_fig.add_trace(go.Scatter(x=portfolio_df['Asset'], y=portfolio_df['Value'], mode='markers', marker=dict(size=portfolio_df['Weight'], sizemode='diameter'), text=portfolio_df['Asset'], name=backtester.strategy_name))

    constituents_fig.update_layout(
        title="Portfolio Constituents",
        xaxis_title="Asset",
        yaxis_title="Value",
        showlegend=True,
        width=plot_width,  # Set the width
        height=plot_height # Set the height
    )

    constituents_fig.update_traces(textposition='top center')
    st.plotly_chart(constituents_fig)




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

json_data_list = []
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

    json_file_name = f"{full_strategy}.json"
    json_data = load_json_data(json_file_name)
    json_data_list.append(json_data)

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

json_data = load_json_data(json_file_name)
display_json_data(json_data_list, backtesters)