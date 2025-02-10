import streamlit as st
import pandas as pd
from process_data import get_processed_data
import UI
from process_data import process_onehot_norm, train_model
# Load processed data
df,df_test = get_processed_data()

# Normalization and one-hot encoding
concat_df= process_onehot_norm(df, df_test)

# Train model
train_model(concat_df)

# UI interface
UI.run_ui(df)
