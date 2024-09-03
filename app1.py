import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import shap
from sklearn.inspection import plot_partial_dependence
import networkx as nx
import sqlalchemy
import pymongo
import requests
import json
import os
from io import StringIO
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import altair as alt
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Advanced Data Analyzer")

# Custom CSS to improve the UI
def apply_custom_theme(theme):
    if theme == "Light":
        st.markdown("""
        <style>
            .reportview-container {background: #f0f2f6;}
            .sidebar .sidebar-content {background: #ffffff;}
            .Widget>label {color: #31333F; font-weight: bold;}
            .stButton>button {color: #ffffff; background-color: #0068c9; border-radius: 5px;}
        </style>
        """, unsafe_allow_html=True)
    elif theme == "Dark":
        st.markdown("""
        <style>
            .reportview-container {background: #1e1e1e; color: #cfcfcf;}
            .sidebar .sidebar-content {background: #333333;}
            .Widget>label {color: #cfcfcf; font-weight: bold;}
            .stButton>button {color: #ffffff; background-color: #d33f3f; border-radius: 5px;}
        </style>
        """, unsafe_allow_html=True)

# Data Source Integration

def connect_to_database(db_type, host, port, username, password, database):
    try:
        port = int(port) if port else None
    except ValueError:
        st.error("Invalid port number. Please enter a valid integer.")
        return None

    if db_type == 'MySQL':
        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif db_type == 'PostgreSQL':
        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    else:
        st.error("Unsupported database type")
        return None
    
    engine = sqlalchemy.create_engine(connection_string)
    return engine

def connect_to_mongodb(host, port, username, password, database):
    client = pymongo.MongoClient(f"mongodb://{username}:{password}@{host}:{port}/{database}")
    return client[database]

def connect_to_google_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive.readonly'])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            st.error("Google Drive authentication failed. Please set up credentials.")
            return None
    
    service = build('drive', 'v3', credentials=creds)
    return service

def fetch_api_data(api_url, params=None, headers=None):
    response = requests.get(api_url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API request failed with status code {response.status_code}")
        return None

def load_data_from_source(source_type):
    if source_type == "File Upload":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
    elif source_type == "Database":
        db_type = st.selectbox("Select database type", ["MySQL", "PostgreSQL", "MongoDB"])
        host = st.text_input("Host")
        port = st.text_input("Port")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database name")
        
        if db_type in ["MySQL", "PostgreSQL"]:
            engine = connect_to_database(db_type, host, port, username, password, database)
            if engine:
                query = st.text_area("Enter your SQL query")
                if st.button("Execute Query"):
                    return pd.read_sql(query, engine)
        elif db_type == "MongoDB":
            db = connect_to_mongodb(host, port, username, password, database)
            if db:
                collection = st.text_input("Enter collection name")
                if collection:
                    return pd.DataFrame(list(db[collection].find()))
    elif source_type == "Cloud Storage":
        service = connect_to_google_drive()
        if service:
            file_id = st.text_input("Enter Google Drive file ID")
            if file_id:
                request = service.files().get_media(fileId=file_id)
                file_content = request.execute()
                return pd.read_csv(StringIO(file_content.decode('utf-8')))
    elif source_type == "API":
        api_url = st.text_input("Enter API URL")
        if api_url:
            data = fetch_api_data(api_url)
            if data:
                return pd.DataFrame(data)
    
    return None

# Advanced Data Preprocessing

def detect_data_types(data):
    return data.dtypes

def handle_missing_data(data):
    strategy = st.selectbox("Select imputation strategy", ["Mean", "Median", "Most Frequent", "KNN"])
    if strategy == "KNN":
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=strategy.lower())
    
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

def handle_outliers(data):
    method = st.selectbox("Select outlier handling method", ["Z-Score", "IQR"])
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    if method == "Z-Score":
        z_scores = np.abs(stats.zscore(data[numeric_columns]))
        data = data[(z_scores < 3).all(axis=1)]
    elif method == "IQR":
        Q1 = data[numeric_columns].quantile(0.25)
        Q3 = data[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return data

def engineer_features(data):
    st.subheader("Feature Engineering")
    
    # Polynomial features
    if st.checkbox("Create polynomial features"):
        degree = st.slider("Select degree", 2, 5)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            for d in range(2, degree + 1):
                data[f"{col}^{d}"] = data[col] ** d
    
    # Interaction features
    if st.checkbox("Create interaction features"):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col1, col2 = numeric_columns[i], numeric_columns[j]
                data[f"{col1}*{col2}"] = data[col1] * data[col2]
    
    # Binning
    if st.checkbox("Perform binning"):
        column_to_bin = st.selectbox("Select column to bin", data.columns)
        num_bins = st.slider("Number of bins", 2, 10)
        data[f"{column_to_bin}_binned"] = pd.cut(data[column_to_bin], bins=num_bins, labels=range(num_bins))
    
    return data

def preprocess_data(data):
    st.subheader("Data Preprocessing")
    
    if st.checkbox("Handle missing data"):
        data = handle_missing_data(data)
    
    if st.checkbox("Handle outliers"):
        data = handle_outliers(data)
    
    if st.checkbox("Normalize numerical features"):
        scaler = StandardScaler()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    if st.checkbox("Encode categorical variables"):
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = LabelEncoder().fit_transform(data[col])
    
    if st.checkbox("Perform feature engineering"):
        data = engineer_features(data)
    
    return data

# Enhanced Visualizations

def create_visualization(data):
    st.subheader("Basic Charts")
    chart_type = st.selectbox("Select chart type", 
                              ['Line Chart', 'Bar Chart', 'Scatter Plot', 'Box Plot', 'Violin Plot', 
                               'Histogram', 'Heatmap', '3D Scatter', 'Parallel Coordinates', 'Sunburst'])
    
    x_column = st.selectbox("Select X-axis column", data.columns)
    y_column = st.selectbox("Select Y-axis column", data.columns)
    color_column = st.selectbox("Select color column (optional)", [None] + list(data.columns))
    
    if chart_type == 'Line Chart':
        fig = px.line(data, x=x_column, y=y_column, color=color_column)
    elif chart_type == 'Bar Chart':
        fig = px.bar(data, x=x_column, y=y_column, color=color_column)
    elif chart_type == 'Scatter Plot':
        fig = px.scatter(data, x=x_column, y=y_column, color=color_column)
    elif chart_type == 'Box Plot':
        fig = px.box(data, x=x_column, y=y_column, color=color_column)
    elif chart_type == 'Violin Plot':
        fig = px.violin(data, x=x_column, y=y_column, color=color_column)
    elif chart_type == 'Histogram':
        fig = px.histogram(data, x=x_column, color=color_column)
    elif chart_type == 'Heatmap':
        fig = px.imshow(data.corr())
    elif chart_type == '3D Scatter':
        z_column = st.selectbox("Select Z-axis column", data.columns)
        fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, color=color_column)
    elif chart_type == 'Parallel Coordinates':
        fig = px.parallel_coordinates(data, color=color_column)
    elif chart_type == 'Sunburst':
        fig = px.sunburst(data, path=[x_column, y_column], values=color_column)
    
    st.plotly_chart(fig)

def create_dashboard(data):
    st.subheader("Interactive Dashboard")
    
    # Allow user to select columns for the dashboard
    dashboard_columns = st.multiselect("Select columns for the dashboard", data.columns)
    
    if dashboard_columns:
        # Create a 2x2 grid of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            st.subheader("Scatter Plot")
            x_scatter = st.selectbox("X-axis (Scatter)", dashboard_columns)
            y_scatter = st.selectbox("Y-axis (Scatter)", dashboard_columns)
            fig_scatter = px.scatter(data, x=x_scatter, y=y_scatter)
            st.plotly_chart(fig_scatter)
            
            # Histogram
            st.subheader("Histogram")
            hist_column = st.selectbox("Column (Histogram)", dashboard_columns)
            fig_hist = px.histogram(data, x=hist_column)
            st.plotly_chart(fig_hist)
        
        with col2:
            # Box plot
            st.subheader("Box Plot")
            box_column = st.selectbox("Column (Box Plot)", dashboard_columns)
            fig_box = px.box(data, y=box_column)
            st.plotly_chart(fig_box)
            
            # Line chart
            st.subheader("Line Chart")
            x_line = st.selectbox("X-axis (Line)", dashboard_columns)
            y_line = st.selectbox("Y-axis (Line)", dashboard_columns)
            fig_line = px.line(data, x=x_line, y=y_line)
            st.plotly_chart(fig_line)

def create_map(data):
    st.subheader("Geospatial Visualization")
    
    lat_column = st.selectbox("Select latitude column", data.columns)
    lon_column = st.selectbox("Select longitude column", data.columns)
    color_column = st.selectbox("Select color column (optional)", [None] + list(data.columns))
    
    fig = px.scatter_mapbox(data, lat=lat_column, lon=lon_column, color=color_column,
                            zoom=3, height=600)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

def create_time_series(data):
    st.subheader("Time Series Analysis")
    
    date_column = st.selectbox("Select date column", data.columns)
    value_column = st.selectbox("Select value column", data.columns)
    
    if date_column is None or value_column is None:
        st.warning("Please select valid columns for date and value.")
        return
    
    # Convert date column to datetime
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort data by date
    data = data.sort_values(date_column)
    
    # Create time series plot
    fig = px.line(data, x=date_column, y=value_column)
    st.plotly_chart(fig)
    
    # Simple forecasting
    if st.checkbox("Perform simple forecasting"):
        forecast_periods = st.slider("Number of periods to forecast", 1, 100)
        
        # Fit a simple linear regression model
        X = np.arange(len(data)).reshape(-1, 1)
        y = data[value_column].values
        model = LinearRegression().fit(X, y)
        
        # Make predictions
        future_X = np.arange(len(data), len(data) + forecast_periods).reshape(-1, 1)
        future_y = model.predict(future_X)
        
        # Create forecast plot
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=data[date_column], y=data[value_column], name="Actual"))
        fig_forecast.add_trace(go.Scatter(x=pd.date_range(start=data[date_column].iloc[-1], periods=forecast_periods+1, freq='D')[1:],
                                          y=future_y, name="Forecast"))
        st.plotly_chart(fig_forecast)

def create_network_graph(data):
    st.subheader("Network Graph")
    
    source_column = st.selectbox("Select source column", data.columns)
    target_column = st.selectbox("Select target column", data.columns)
    weight_column = st.selectbox("Select weight column (optional)", [None] + list(data.columns))
    
    # Create a network graph
    G = nx.from_pandas_edgelist(data, source=source_column, target=target_column, edge_attr=weight_column)
    
    # Calculate node positions
    pos = nx.spring_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Color node points by the number of connections
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]} - # of connections: {len(adjacencies[1])}')
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    st.plotly_chart(fig)

# Statistical Analysis

def perform_hypothesis_test(data):
    st.subheader("Hypothesis Testing")
    
    test_type = st.selectbox("Select test type", ["T-Test", "Chi-Square Test", "ANOVA"])
    
    if test_type == "T-Test":
        column1 = st.selectbox("Select first column", data.columns)
        column2 = st.selectbox("Select second column", data.columns)
        
        result = stats.ttest_ind(data[column1], data[column2])
        st.write(f"T-Statistic: {result.statistic}")
        st.write(f"P-Value: {result.pvalue}")
    
    elif test_type == "Chi-Square Test":
        column1 = st.selectbox("Select first categorical column", data.select_dtypes(include=['object']).columns)
        column2 = st.selectbox("Select second categorical column", data.select_dtypes(include=['object']).columns)
        
        contingency_table = pd.crosstab(data[column1], data[column2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        st.write(f"Chi-Square Statistic: {chi2}")
        st.write(f"P-Value: {p_value}")
    
    elif test_type == "ANOVA":
        dependent_var = st.selectbox("Select dependent variable", data.select_dtypes(include=[np.number]).columns)
        independent_var = st.selectbox("Select independent variable", data.select_dtypes(include=['object']).columns)
        
        # Ensure valid columns are selected
        if dependent_var is not None and independent_var is not None:
            groups = data.groupby(independent_var)[dependent_var].apply(list).values
            f_value, p_value = stats.f_oneway(*groups)
            st.write(f"F-value: {f_value}")
            st.write(f"p-value: {p_value}")

def perform_regression(data):
    st.subheader("Regression Analysis")
    
    target = st.selectbox("Select target variable", data.columns)
    features = st.multiselect("Select features", [col for col in data.columns if col != target])
    
    if features:
        X = data[features]
        y = data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        st.write("Model Coefficients:")
        for feature, coef in zip(features, model.coef_):
            st.write(f"{feature}: {coef}")
        
        st.write(f"R-squared: {r2_score(y_test, y_pred)}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

def perform_anova(data):
    st.subheader("ANOVA")
    
    dependent_var = st.selectbox("Select dependent variable", data.select_dtypes(include=[np.number]).columns)
    independent_var = st.selectbox("Select independent variable", data.select_dtypes(include=['object']).columns)
    
    groups = data.groupby(independent_var)[dependent_var].apply(list).values
    f_value, p_value = stats.f_oneway(*groups)
    
    st.write(f"F-value: {f_value}")
    st.write(f"p-value: {p_value}")

def perform_factor_analysis(data):
    st.subheader("Factor Analysis")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    selected_columns = st.multiselect("Select columns for factor analysis", numeric_columns)
    
    if selected_columns:
        n_factors = st.slider("Number of factors", 1, len(selected_columns))
        
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa_result = fa.fit_transform(data[selected_columns])
        
        loadings = pd.DataFrame(fa.components_.T, columns=[f'Factor{i+1}' for i in range(n_factors)], index=selected_columns)
        st.write("Factor Loadings:")
        st.write(loadings)
        
        fig = px.imshow(loadings, aspect="auto", labels=dict(color="Loading"))
        st.plotly_chart(fig)

# Machine Learning Integration

def perform_automl(data):
    st.subheader("AutoML")
    
    target = st.selectbox("Select target variable", data.columns)
    features = [col for col in data.columns if col != target]
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if data[target].dtype == 'object':
        # Classification
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
    else:
        # Regression
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")
    
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    st.write("Feature Importance:")
    st.write(feature_importance)
    
    fig = px.bar(feature_importance, x='feature', y='importance')
    st.plotly_chart(fig)

def perform_clustering(data):
    st.subheader("Clustering")
    
    algorithm = st.selectbox("Select clustering algorithm", ["K-Means", "DBSCAN", "Hierarchical Clustering"])
    features = st.multiselect("Select features for clustering", data.columns)
    
    if features and len(features) >= 2:
        X = data[features]
        
        if algorithm == "K-Means":
            n_clusters = st.slider("Number of clusters", 2, 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
        elif algorithm == "DBSCAN":
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
            min_samples = st.slider("Minimum samples", 2, 10)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X)
        elif algorithm == "Hierarchical Clustering":
            n_clusters = st.slider("Number of clusters", 2, 10)
            linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            clusters = hierarchical.fit_predict(X)
        
        data['Cluster'] = clusters
        
        fig = px.scatter(data, x=features[0], y=features[1], color='Cluster')
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least two features for clustering.")

def build_classification_model(data):
    st.subheader("Classification Model")
    
    target = st.selectbox("Select target variable", data.select_dtypes(include=['object']).columns)
    if target:
        features = [col for col in data.columns if col != target]
        
        X = data[features]
        y = data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"Accuracy: {accuracy}")
        
        feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        st.write("Feature Importance:")
        st.write(feature_importance)
        
        fig = px.bar(feature_importance, x='feature', y='importance')
        st.plotly_chart(fig)

        # SHAP Values for Explainability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        st.subheader("SHAP Values")
        shap.summary_plot(shap_values[1], X_test, plot_type="bar")
        st.pyplot()

        # Partial Dependence Plot
        st.subheader("Partial Dependence Plot")
        plot_partial_dependence(model, X_test, features)
        st.pyplot()

def build_regression_model(data):
    st.subheader("Regression Model")
    
    target = st.selectbox("Select target variable", data.select_dtypes(include=[np.number]).columns)
    features = [col for col in data.columns if col != target]
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")
    
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    st.write("Feature Importance:")
    st.write(feature_importance)
    
    fig = px.bar(feature_importance, x='feature', y='importance')
    st.plotly_chart(fig)

    # SHAP Values for Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    st.subheader("SHAP Values")
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    st.pyplot()

    # Partial Dependence Plot
    st.subheader("Partial Dependence Plot")
    plot_partial_dependence(model, X_test, features)
    st.pyplot()

def analyze_feature_importance(data):
    st.subheader("Feature Importance Analysis")
    
    target = st.selectbox("Select target variable", data.columns)
    features = [col for col in data.columns if col != target]
    
    X = data[features]
    y = data[target]
    
    if data[target].dtype == 'object':
        # Classification
        model = RandomForestClassifier(random_state=42)
    else:
        # Regression
        model = RandomForestRegressor(random_state=42)
    
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    st.write("Feature Importance:")
    st.write(feature_importance)
    
    fig = px.bar(feature_importance, x='feature', y='importance')
    st.plotly_chart(fig)

def perform_text_analysis(data):
    st.subheader("Natural Language Processing")

    text_column = st.selectbox("Select text column", data.columns)
    if text_column:
        # Ensure the selected column contains text data
        if data[text_column].dtype == object:
            st.subheader("Text Analysis Tools")
            if st.checkbox("Perform Sentiment Analysis"):
                # Sentiment Analysis Placeholder
                st.write("Sentiment Analysis performed here")

            if st.checkbox("Perform Topic Modeling"):
                n_topics = st.slider("Number of topics", 2, 10)
                vectorizer = TfidfVectorizer(stop_words='english')
                lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

                pipeline = Pipeline([
                    ('tfidf', vectorizer),
                    ('lda', lda)
                ])

                pipeline.fit(data[text_column])
                st.write("Topic Modeling performed here")
        else:
            st.warning("Please select a valid text column for NLP tasks.")

def guided_analytics(data):
    st.subheader("Guided Analytics")
    st.write("Step-by-step explanation of the analysis process will be presented here.")

def generate_report(data):
    st.subheader("Automated Insight Generation")
    st.write("Narrative writing and key findings will be generated here.")

def user_authentication():
    st.subheader("User Authentication")
    st.write("Implement user authentication and role-based access control here.")

def collaboration_features():
    st.subheader("Collaboration Features")
    st.write("Sharing, commenting, and version control features will be added here.")

def export_reporting(data):
    st.subheader("Export and Reporting")
    
    if st.checkbox("Export visualizations"):
        export_format = st.selectbox("Select export format", ["PNG", "SVG", "PDF"])
        st.write(f"Exporting visualizations as {export_format}")

    if st.checkbox("Generate automated reports"):
        st.write("Generating automated report with key findings...")

def google_gemini_api(text):
    api_key = "your_gemini_api"
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": text
                    }
                ]
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API request failed with status code {response.status_code}")
        return None

def ai_powered_analysis(data):
    st.subheader("AI-Powered Analysis with Google Gemini")
    text_input = st.text_area("Enter text for analysis or content generation")
    
    if text_input and st.button("Analyze with AI"):
        ai_response = google_gemini_api(text_input)
        if ai_response:
            st.write(ai_response)
            st.write("AI-Powered Analysis completed.")

@lru_cache(maxsize=32)
def load_and_cache_data(source_type):
    return load_data_from_source(source_type)

def main():
    st.title("Advanced Data Analyzer")

    # Sidebar for themes and layouts
    with st.sidebar:
        st.header("Customization")
        theme = st.selectbox("Select Theme", ["Light", "Dark"])
        apply_custom_theme(theme)
        st.header("Data Source")
        source_type = st.selectbox("Select data source", ["File Upload", "Database", "Cloud Storage", "API"])
        
        data = load_and_cache_data(source_type)
        
        if data is not None:
            st.success("Data loaded successfully!")
            st.write("Data Shape:", data.shape)
            
            if st.checkbox("Show raw data"):
                st.write(data)
            
            if st.checkbox("Preprocess data"):
                data = preprocess_data(data)
                st.success("Data preprocessed!")

    # Main content area
    if 'data' in locals() and data is not None:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "Data Overview", "Visualization", "Statistical Analysis", "Machine Learning", 
            "NLP", "Guided Analytics", "Collaboration", "Export & Reporting", "AI-Powered Features", "Custom Scripting"])
        
        with tab1:
            st.header("Data Overview")
            st.write("Data Types:")
            st.write(detect_data_types(data))
            
            if st.checkbox("Show summary statistics"):
                st.write(data.describe())
            
            if st.checkbox("Show correlation matrix"):
                fig = px.imshow(data.corr())
                st.plotly_chart(fig)

        with tab2:
            st.header("Data Visualization")
            viz_type = st.selectbox("Select visualization type", 
                                    ["Basic Charts", "Interactive Dashboard", "Geospatial", "Time Series", "Network Graph"])
            
            if viz_type == "Basic Charts":
                create_visualization(data)
            elif viz_type == "Interactive Dashboard":
                create_dashboard(data)
            elif viz_type == "Geospatial":
                create_map(data)
            elif viz_type == "Time Series":
                create_time_series(data)
            elif viz_type == "Network Graph":
                create_network_graph(data)

        with tab3:
            st.header("Statistical Analysis")
            analysis_type = st.selectbox("Select analysis type", 
                                         ["Hypothesis Testing", "Regression Analysis", "ANOVA", "Factor Analysis"])
            
            if analysis_type == "Hypothesis Testing":
                perform_hypothesis_test(data)
            elif analysis_type == "Regression Analysis":
                perform_regression(data)
            elif analysis_type == "ANOVA":
                perform_anova(data)
            elif analysis_type == "Factor Analysis":
                perform_factor_analysis(data)

        with tab4:
            st.header("Machine Learning")
            ml_task = st.selectbox("Select machine learning task", 
                                   ["AutoML", "Clustering", "Classification", "Regression", "Feature Importance"])
            
            if ml_task == "AutoML":
                perform_automl(data)
            elif ml_task == "Clustering":
                perform_clustering(data)
            elif ml_task == "Classification":
                build_classification_model(data)
            elif ml_task == "Regression":
                build_regression_model(data)
            elif ml_task == "Feature Importance":
                analyze_feature_importance(data)

        with tab5:
            st.header("Natural Language Processing")
            perform_text_analysis(data)

        with tab6:
            st.header("Guided Analytics")
            guided_analytics(data)

        with tab7:
            st.header("Collaboration Features")
            collaboration_features()

        with tab8:
            st.header("Export and Reporting")
            export_reporting(data)

        with tab9:
            st.header("AI-Powered Features")
            ai_powered_analysis(data)

        with tab10:
            st.header("Custom Scripting")
            st.text_area("Write and execute custom Python scripts here")

        # Data download option
        st.download_button(
            label="Download processed data as CSV",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name="processed_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
