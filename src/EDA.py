import streamlit as st
from process_data import divide_num_cat_cols
import matplotlib.pyplot as plt
def show(df):

    # Layout for better visualization
    c1,c2,c3=st.columns([1,10,1])

    # Exctract numerical and categorical columns
    numcols,catcols,cont_cols,discrete_cols=divide_num_cat_cols(df)
    
    st.write('\n')
    # Relationship with Discrete Features
    
    # Plot discrete-features vs SalePrice
    for feature in discrete_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        df.groupby(feature)['SalePrice'].median().plot()
        ax.set_xlabel(feature)
        ax.set_ylabel('Sales Price')
        ax.set_title(f'{feature} vs Sale Price')
        
        with c2:
            st.pyplot(fig)

    # Relationship with Continuous Features

    # Plot histograms for continuous features
    for feature in cont_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Histogram of {feature}', fontsize=14)
        plt.grid(True)
        
        with c2:
            st.pyplot(fig)

    # Relationship with Categorical Features
    
    # Plot categorical features vs SalePrice 
    for feature in catcols:
        fig, ax = plt.subplots(figsize=(8, 5))
        df.groupby(feature)['SalePrice'].median().plot(
            kind='bar', ax=ax, color='skyblue', edgecolor='black'
        )
        ax.set_title(f'{feature} vs Median SalePrice', fontsize=14)
        plt.xticks()

        with c2:
            st.pyplot(fig)
