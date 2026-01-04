import streamlit as st
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

# Clear cache on app startup
st.cache_data.clear()
st.cache_resource.clear()

# Page config
st.set_page_config(page_title="IRIS Dataset EDA", layout="wide")

# Title
st.title("🌸 IRIS Dataset – Exploratory Data Analysis")

# Load dataset
@st.cache_data
def load_data():
    return sns.load_dataset("iris")

data = load_data()

# Sidebar
st.sidebar.header("EDA Options")
eda_option = st.sidebar.selectbox(
    "Select Analysis Type",
    [
        "Dataset Overview",
        "Statistical Summary",
        "Distribution Plot",
        "Joint Plot",
        "Pair Plot",
        "Boxen Plot",
        "Strip Plot",
        "Swarm Plot"
    ]
)

# ===================== DATASET OVERVIEW =====================
if eda_option == "Dataset Overview":
    st.subheader("📄 Dataset Preview")
    st.dataframe(data)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Missing Values", data.isnull().sum().sum())
    
    st.subheader("Column Info")
    st.write(data.dtypes)

# ===================== STATISTICS =====================
elif eda_option == "Statistical Summary":
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(data.describe())

# ===================== DISTRIBUTION =====================
elif eda_option == "Distribution Plot":
    st.subheader("📈 Distribution Plot")
    column = st.selectbox(
        "Select Column",
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    
    fig, ax = plt.subplots()
    sns.histplot(data[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)
    plt.close()

# ===================== JOINT PLOT =====================
elif eda_option == "Joint Plot":
    st.subheader("🔗 Joint Plot")
    
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    x_col = st.selectbox("X Axis", numeric_cols, key="joint_x")
    y_col = st.selectbox("Y Axis", numeric_cols, index=1, key="joint_y")
    
    if x_col == y_col:
        st.warning("⚠️ Please select different columns for X and Y axis")
    else:
        kind = st.selectbox("Plot Type", ["scatter", "reg", "hex", "kde"])
        
        # Joint plot doesn't support hue for all kinds
        if kind in ["scatter", "kde"]:
            fig = sns.jointplot(x=x_col, y=y_col, data=data, kind=kind, hue="species")
        else:
            fig = sns.jointplot(x=x_col, y=y_col, data=data, kind=kind)
        
        st.pyplot(fig)
        plt.close()

# ===================== PAIR PLOT =====================
elif eda_option == "Pair Plot":
    st.subheader("🔀 Pair Plot (Feature Relationships)")
    st.info("Color coded by species")
    fig = sns.pairplot(data, hue="species")
    st.pyplot(fig)
    plt.close()

# ===================== BOXEN PLOT =====================
elif eda_option == "Boxen Plot":
    st.subheader("📦 Boxen Plot")
    
    y_col = st.selectbox("Select Feature", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxenplot(x="species", y=y_col, data=data, ax=ax)
    ax.set_title(f"Boxen Plot: {y_col} by Species")
    st.pyplot(fig)
    plt.close()

# ===================== STRIP PLOT =====================
elif eda_option == "Strip Plot":
    st.subheader("📌 Strip Plot")
    
    y_col = st.selectbox("Select Feature", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.stripplot(x="species", y=y_col, data=data, ax=ax)
    ax.set_title(f"Strip Plot: {y_col} by Species")
    st.pyplot(fig)
    plt.close()

# ===================== SWARM PLOT =====================
elif eda_option == "Swarm Plot":
    st.subheader("🐝 Swarm Plot")
    
    y_col = st.selectbox("Select Feature", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.swarmplot(x="species", y=y_col, data=data, ax=ax)
    ax.set_title(f"Swarm Plot: {y_col} by Species")
    st.pyplot(fig)
    plt.close()

# Footer
st.markdown("---")
st.markdown("✅ **IRIS Dataset EDA using Streamlit & Seaborn**")
