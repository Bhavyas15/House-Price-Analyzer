import streamlit as st
import numpy as np

from process_data import load_data, preprocess_data, train_model, process_onehot_norm
from collect_input import collect_info
from EDA import show

option = st.sidebar.radio('Select an option:', ['Price Prediction', 'Exploratory Data Analysis', 'Information'])

df,df_test=load_data()
df=preprocess_data(df)

if option=='Price Prediction':
    df_test=preprocess_data(df_test)
    concat_df,concat_arr=process_onehot_norm(df,df_test)

    train_model(concat_df)
    st.title('House Price Prediction 🏡')
    output=collect_info(df)
    st.header(f'Predicted Price : $ {np.round(output[0])}')

elif option=='Information':
    st.header('Information of Parameters \n\n')
    with open("data/data_description.txt", "r") as file:
        content=file.read()
    st.write(content)

else:
    show(df)
    