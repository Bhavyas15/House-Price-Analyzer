# import streamlit as st
# import numpy as np
# import pandas as pd

# from collect_input import collect_info, predict
# from EDA import show

# # Load processed data
# @st.cache_data
# def load_processed_data():
#     return pd.read_csv("data/processed_data.csv")

# df = load_processed_data()

# # Sidebar
# option = st.sidebar.radio(
#     'Select an option:', 
#     ['Price Prediction', 'Exploratory Data Analysis', 'Information'],
#     index=0
# )

# st.title('House Price Prediction 🏡')

# if option=='Price Prediction':

#     # Collect info and processes inputs in the form of dataframe
#     inputs=collect_info(df)

#     # Predict output
#     output=predict(inputs)
    
#     # Button to display predicted price
#     if st.button('Predict Price'):
#         st.header(f'Predicted Price : $ {np.round(output[0])}')

# elif option=='Information':
#     # Display info about parameters
#     st.header('Information of Parameters \n\n')

#     with open("data/data_description.txt", "r") as file:
#         content=file.read() 
#     st.write(content)

# else:
#     # Show EDA
#     show(df)

import streamlit as st
import numpy as np
import pandas as pd
from collect_input import collect_info, predict
from EDA import show

def run_ui(df):
    """Function to render the UI in Streamlit"""
    st.title('House Price Prediction 🏡')

    # Sidebar options with default selection
    option = st.sidebar.radio(
        'Select an option:', 
        ['Price Prediction', 'Exploratory Data Analysis', 'Information'], 
        index=0  # Default selection set to 'Price Prediction'
    )

    if option == 'Price Prediction':
        # Collect user inputs and process
        inputs = collect_info(df)

        # Predict house price
        output = predict(inputs)

        # Display predicted price
        if st.button('Predict Price'):
            st.header(f'Predicted Price: $ {np.round(output[0])}')

    elif option == 'Information':
        # Display dataset parameter details
        st.header('Information of Parameters')

        with open("data/data_description.txt", "r") as file:
            content = file.read()
        st.write(content)

    else:
        # Show Exploratory Data Analysis
        show(df)
