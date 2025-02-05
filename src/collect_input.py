import streamlit as st
import numpy as np
import pandas as pd
import pickle
from process_data import divide_num_cat_cols,category_onehot

with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

def collect_info(df):
    numcols,catcols,numcols_cont,numcols_discrete=divide_num_cat_cols(df)
    dict={}
    for feature in numcols_discrete:
        val=st.number_input(f'Select {feature}', df[feature].min(),df[feature].max(),step=1,value=int(df[feature].mode()[0]))
        dict[feature]=val
    for feature in numcols_cont:
        val=st.slider(f'{feature}',np.round(int(df[feature].min())),np.round(int(df[feature].max())),value=np.round(int(df[feature].mode()[0])))
        dict[feature]=val

    col1,col2,col3=st.columns([1,1,1])
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

    cat_onehot_df=category_onehot(catcols,df)
    cat_onehot_input_df=cat_onehot_df.drop(['SalePrice'],axis=1)
    b=pd.DataFrame(columns=cat_onehot_input_df.columns[11:])
    c=pd.DataFrame(columns=cat_onehot_input_df.columns[:11])
    new_row = {col: 1 if col in cat_inp else 0 for col in b.columns}
    b = pd.concat([b, pd.DataFrame([new_row])], ignore_index=True)
    
    inputs_num=pd.DataFrame([[dict['LotFrontage'],dict['LotArea'],dict['YearBuilt'],dict['MasVnrArea'],
            dict['TotalBsmtSF'],dict['GrLivArea'],dict['Fireplaces'],dict['GarageCars'],dict['WoodDeckSF'],dict['TotBath'],dict['TotalPorchSF']]], 
            columns=cat_onehot_input_df.columns[:11])
    inputs=pd.concat([inputs_num,b], axis=1)
    inputs_scaled=scaler.transform(inputs)
    output=model.predict(inputs_scaled)
    
    return output
    
