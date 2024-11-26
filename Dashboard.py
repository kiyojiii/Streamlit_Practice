import random
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#######################################
# PAGE SETUP
#######################################

st.set_page_config(page_title="Dashboard", page_icon=":bar_chart:", layout="wide")

st.title("Resampling and Performance Metrics Dashboard")
st.markdown("_Laboratory Exercise #2_")

with st.sidebar:
    st.header("Upload a CSV File")
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.info(" Upload a file through config", icon="ℹ️")
    st.stop()

#######################################
# DATA LOADING
#######################################


@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df


df = load_data(uploaded_file)

# Check if the CSV is lung cancer or air quality
csv_type = 'lung cancer' if 'LUNG_CANCER' in df.columns else 'air quality'

all_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

with st.expander("Data Preview"):
    st.dataframe(df)

#######################################
# VISUALIZATION METHODS
#######################################

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={"color": color_graph},
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound):
    """
    Create and display a gauge chart using Plotly for a metric.
    """
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": indicator_suffix, "font.size": 18},
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={"text": indicator_title, "font": {"size": 20}},
        )
    )
    fig.update_layout(
        height=250,  # Adjusted height for better appearance
        margin=dict(l=10, r=10, t=30, b=10, pad=5),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_top_right():
    if csv_type == 'lung cancer':
        # Prepare data for clustered bar chart
        gender_lung_cancer_data = df.groupby(['GENDER', 'LUNG_CANCER']).size().reset_index(name='Count')

        # Define custom colors for 'YES' and 'NO'
        custom_colors = {
            "YES": "#ff0e0e",  # Tomato Red for Lung Cancer 'YES'
            "NO": "#4BB543",   # Steel Blue for Lung Cancer 'NO'
        }

        # Clustered Bar Graph for Gender and Lung Cancer
        fig_gender_lung_cancer = px.bar(
            gender_lung_cancer_data,
            x='GENDER',
            y='Count',
            color='LUNG_CANCER',
            barmode='group',
            labels={'GENDER': 'Gender', 'Count': 'Count', 'LUNG_CANCER': 'Lung Cancer'},
            title="Gender Distribution vs Lung Cancer (Yes/No)",
            color_discrete_map=custom_colors,  # Apply custom color mapping
        )
        st.plotly_chart(fig_gender_lung_cancer, use_container_width=True)

    elif csv_type == 'air quality':
        # CO Level Distribution Bar Graph
        co_level_order = ["Very Low", "Low", "Moderate", "High", "Very High"]  # Define the correct order
        co_level_colors = {
            "Very Low": "#00FF00",  # Green
            "Low": "#7FFF00",       # Chartreuse
            "Moderate": "#FFFF00",  # Yellow
            "High": "#FFA500",      # Orange
            "Very High": "#FF0000", # Red
        }

        # Ensure CO_level is a categorical column with the specified order
        df['CO_level'] = pd.Categorical(df['CO_level'], categories=co_level_order, ordered=True)

        # Calculate CO Level counts
        co_level_count = df['CO_level'].value_counts().reindex(co_level_order)

        # Plot the bar chart with custom colors
        fig = px.bar(
            x=co_level_count.index,
            y=co_level_count.values,
            labels={'x': 'CO Level', 'y': 'Count'},
            title="CO Level Distribution",
            color=co_level_count.index,
            color_discrete_map=co_level_colors,  # Apply custom color mapping
        )

        # Update layout to remove legend and maintain the desired order
        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': co_level_order},
            showlegend=False  # Remove the legend
        )

        st.plotly_chart(fig, use_container_width=True)

def plot_bottom_left():
    if csv_type == 'lung cancer':
        # Group ages into bins
        age_bins = pd.cut(df['AGE'], bins=range(0, 101, 10), right=False, labels=[f"{i}-{i+9}" for i in range(0, 100, 10)])
        df['Age Group'] = age_bins

        # Calculate lung cancer rate per age group
        age_group_lung_cancer = (
            df.groupby('Age Group')['LUNG_CANCER']
            .apply(lambda x: (x == "YES").mean())  # Calculate percentage of "YES"
            .reset_index()
            .rename(columns={'LUNG_CANCER': 'Lung Cancer Rate'})
        )

        # Line graph for Lung Cancer Rate by Age Group
        fig = px.line(
            age_group_lung_cancer,
            x='Age Group',
            y='Lung Cancer Rate',
            markers=True,
            title="Lung Cancer Rate by Age Group",
            labels={'Lung Cancer Rate': 'Lung Cancer Rate (%)'},
        )
        st.plotly_chart(fig, use_container_width=True)

    elif csv_type == 'air quality':
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter out rows where T equals -200
        filtered_df = df[df['T'] != -200]

        # Aggregate data to reduce clutter (daily average of T)
        daily_avg = filtered_df.groupby('Date')['T'].mean().reset_index()

        # Plot the trend of daily average T over time
        fig = px.line(
            daily_avg,
            x="Date",
            y="T",
            markers=True,
            title="Daily Average Temperature Trend Over Time",
            labels={"T": "Temperature (T)", "Date": "Date"},
        )
        # Update line and marker colors to orange
        fig.update_traces(line_color='orange', marker_color='orange')

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Temperature (T)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_bottom_right():
    if csv_type == 'lung cancer':
        # Define custom colors for 'Male' and 'Female'
        gender_colors = {
            "M": "#4682B4",  # Blue for Male
            "F": "#FF69B4",  # Pink for Female
        }

        # Stacked Bar Chart for Gender, Age, and Lung Cancer
        fig = px.histogram(
            df,
            x="AGE",
            color="GENDER",
            barmode="stack",
            title="Gender and Age Distribution for Lung Cancer",
            color_discrete_map=gender_colors,  # Apply custom color mapping
        )
        st.plotly_chart(fig, use_container_width=True)

    elif csv_type == 'air quality':
        # Stacked Bar Chart for Date, Time and CO Level
        fig = px.histogram(df, x="Date", color="CO_level", barmode="stack", title="Date and CO Level Distribution")
        st.plotly_chart(fig, use_container_width=True)


#######################################
# STREAMLIT LAYOUT
#######################################

# Column Layout
top_left_column, top_right_column = st.columns((2, 1))
bottom_left_column, bottom_right_column = st.columns(2)

with top_left_column:
    column_1, column_2, column_3, column_4 = st.columns(4)

    # Top-left column: Row 1 - Display Counts
    with column_1:
        # Data Count
        plot_metric("Data Count", len(df), show_graph=False)

    with column_2:
        # Data Columns
        plot_metric("Column Count", len(df.columns), show_graph=False)

    with column_3:
        # Missing Data Count
        missing_data_count = df.isnull().sum().sum()
        plot_metric("Missing Data", missing_data_count, show_graph=False)

    with column_4:
        # Missing Data Ratio
        unique_data = (len(df.drop_duplicates()) / len(df)) * 100
        plot_metric("Unique Data", unique_data, suffix="%", show_graph=False)

    # Top-left column: Row 2 - Display Gauges
    with column_1:
        # Gauge for Data Count
        plot_gauge(
            indicator_number=len(df),
            indicator_color="#4CAF50",
            indicator_suffix="",
            indicator_title="Data Count",
            max_bound=len(df) * 1.2,  # Dynamic bound
        )

    with column_2:
        # Gauge for Data Columns
        plot_gauge(
            indicator_number=len(df.columns),
            indicator_color="#2196F3",
            indicator_suffix="",
            indicator_title="Data Columns",
            max_bound=len(df.columns) * 1.5,
        )

    with column_3:
        # Gauge for Missing Data Count
        plot_gauge(
            indicator_number=missing_data_count,
            indicator_color="#FF9800",
            indicator_suffix="",
            indicator_title="Missing Data",
            max_bound=(len(df) * len(df.columns)) * 0.2,
        )

    with column_4:
        # Gauge for Missing Data Ratio
        plot_gauge(
            indicator_number=unique_data,
            indicator_color="#E91E63",
            indicator_suffix="%",
            indicator_title="Data Uniqueness",
            max_bound=100,  # Ratio is a percentage, so max is 100
        )


with top_right_column:
    plot_top_right()

with bottom_left_column:
    plot_bottom_left()

with bottom_right_column:
    plot_bottom_right()
