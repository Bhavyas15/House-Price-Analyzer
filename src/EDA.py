import streamlit as st
import matplotlib.pyplot as plt
from process_data import divide_num_cat_cols

def plot_discrete_feature(df, feature, col):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(feature)["SalePrice"].median().plot(ax=ax)
    ax.set_xlabel(feature)
    ax.set_ylabel("Sale Price")
    ax.set_title(f"{feature} vs Sale Price")
    with col:
        st.pyplot(fig)

def plot_continuous_feature(df, feature, col):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[feature], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Histogram of {feature}", fontsize=14)
    plt.grid(True)
    with col:
        st.pyplot(fig)

def plot_categorical_feature(df, feature, col):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(feature)["SalePrice"].median().plot(
        kind="bar", ax=ax, color="skyblue", edgecolor="black"
    )
    ax.set_title(f"{feature} vs Median SalePrice", fontsize=14)
    plt.xticks(rotation=45)
    with col:
        st.pyplot(fig)

def show(df):
    col1, col2, col3 = st.columns([1, 10, 1])
    numcols, catcols, cont_cols, discrete_cols = divide_num_cat_cols(df)
    st.write("\n")
    
    for feature in discrete_cols:
        plot_discrete_feature(df, feature, col2)
    for feature in cont_cols:
        plot_continuous_feature(df, feature, col2)
    for feature in catcols:
        plot_categorical_feature(df, feature, col2)
