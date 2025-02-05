import streamlit as st
from process_data import divide_num_cat_cols
import matplotlib.pyplot as plt
def show(df):
    c1,c2,c3=st.columns([1,5,1])
    numcols,catcols,cont_cols,discrete_cols=divide_num_cat_cols(df)
    st.write('\n')
    st.subheader('Relationship with Discrete Features')
    for feature in discrete_cols:
        fig, ax = plt.subplots(figsize=(8, 5))  # Create a figure
        df.groupby(feature)['SalePrice'].median().plot()
        ax.set_xlabel(feature)
        ax.set_ylabel('Sales Price')
        ax.set_title(f'{feature} vs Sale Price')
        
        st.pyplot(fig)  # Display plot in Streamlit
    st.subheader('Relationship with Continuous Features')
    for feature in cont_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot histogram
        ax.hist(df[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Histogram of {feature}', fontsize=14)
        plt.grid(True)
        st.pyplot(fig)
    st.subheader('Relationship with Categorical Features')
    for feature in catcols:
        # Create a figure for Streamlit
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create bar plot of median SalePrice for each category
        df.groupby(feature)['SalePrice'].median().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        # Customize plot
        ax.set_title(f'{feature} vs Median SalePrice', fontsize=14)
        plt.xticks()  # Rotate x-axis labels for better readability

        # Display plot in Streamlit
        st.pyplot(fig)