import streamlit as st
import numpy as np

from process_data import load_data, preprocess_data, train_model, process_onehot_norm
from collect_input import collect_info
from EDA import show

# Sidebar
option = st.sidebar.radio(
    'Select an option:', 
    ['Price Prediction', 'Exploratory Data Analysis', 'Information']
)

# Load and process main datasets
df,df_test=load_data()
df=preprocess_data(df)
df_test=preprocess_data(df_test)

st.title('House Price Prediction 🏡')

if option=='Price Prediction':
    # Normalization and one-hot encoding
    concat_df,concat_arr=process_onehot_norm(df,df_test)

    # Train model
    train_model(concat_df)

    # Collect info
    output=collect_info(df)

    # Button to display predicted price
    if st.button('Predict Price'):
        st.header(f'Predicted Price : $ {np.round(output[0])}')

elif option=='Information':
    # Display info about parameters
    st.header('Information of Parameters \n\n')

    with open("data/data_description.txt", "r") as file:
        content=file.read() 
    st.write(content)

else:
    # Show EDA
    show(df)
    
