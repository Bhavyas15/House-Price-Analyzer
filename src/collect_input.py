import streamlit as st
import numpy as np
import pandas as pd
import pickle
from process_data import divide_num_cat_cols,category_onehot

# Loading pre-trained scaler and model
@st.cache_resource
def load_scaler():
    with open("data/scaler.pkl", "rb") as f:
        return pickle.load(f)
@st.cache_resource
def load_model():
    with open("data/xgboost_model.pkl", "rb") as f:
        return pickle.load(f)


def collect_info(df):
    # Get different types of columns
    numcols,catcols,numcols_cont,numcols_discrete=divide_num_cat_cols(df)
    input_data={}

    # Collect information of numerical columns
    for feature in numcols_discrete:
        val=st.number_input(f'Select {feature}', df[feature].min(),df[feature].max(),step=1,value=int(df[feature].mode()[0]))
        input_data[feature]=val
    for feature in numcols_cont:
        val=st.slider(f'{feature}',np.round(int(df[feature].min())),np.round(int(df[feature].max())),value=np.round(int(df[feature].mode()[0])))
        input_data[feature]=val

    # Layout for categorical features
    col1,col2,col3=st.columns([2,1,2])
    columns=[col1,col2,col3]
    cat_inp=[]
    i=0

    for feature in catcols:
        if len(df[feature].unique()) <5 :
            sel_input=columns[2].radio(f'Select {feature}',df[feature].unique())
            cat_inp.append(sel_input)
        else:
            sel_input=columns[0].selectbox(f'Select {feature}',df[feature].unique())
            cat_inp.append(sel_input)
        i+=1

    # One hot encoding of categorical features
    cat_onehot_df=category_onehot(catcols,df)
    cat_onehot_input_df=cat_onehot_df.drop(['SalePrice'],axis=1)
    
    # Create empty DataFrame for categorical inputs
    input_df_cat=pd.DataFrame(columns=cat_onehot_input_df.columns[11:])

    # Encode categorical features as one-hot
    new_row = {col: 1 if col in cat_inp else 0 for col in input_df_cat.columns}
    input_df_cat = pd.concat([input_df_cat, pd.DataFrame([new_row])], ignore_index=True)

    # Create numerical input Dataframe
    inputs_num=pd.DataFrame(
        [[
            input_data['LotFrontage'],
            input_data['LotArea'],
            input_data['YearBuilt'],
            input_data['MasVnrArea'],
            input_data['TotalBsmtSF'],
            input_data['GrLivArea'],
            input_data['Fireplaces'],
            input_data['GarageCars'],
            input_data['WoodDeckSF'],
            input_data['TotBath'],
            input_data['TotalPorchSF']
        ]], 
        columns=cat_onehot_input_df.columns[:11])
    
    # Combine numerical and categorical features
    inputs=pd.concat([inputs_num,input_df_cat], axis=1)
    
    # Scale input features
    scaler=load_scaler()
    inputs_scaled=scaler.transform(inputs)
    
    # Predict using trained model
    model=load_model()
    output=model.predict(inputs_scaled)
    
    return output
    
