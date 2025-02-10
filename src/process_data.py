import pandas as pd
import numpy as np
import xgboost
import pickle
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load and process main datasets
@st.cache_data
def get_processed_data():
    df, df_test = load_data()
    # Filling missing values, remove outliers, remove unnecessary columns
    df = preprocess_data(df)
    df_test = preprocess_data(df_test)

    # Save processed data (for use in UI.py)
    df.to_csv("data/processed_data.csv", index=False)
    df_test.to_csv("data/processed_test_data.csv", index=False)

    return df,df_test

@st.cache_data
def load_data():
    df=pd.read_csv('data/train.csv')
    df_test=pd.read_csv('data/test.csv')
    return df,df_test


def preprocess_data(df):
    # Fill Missing Values
    df['LotFrontage'].fillna(np.round(df['LotFrontage'].mean()),inplace=True)
    df['MasVnrType'].fillna('None', inplace=True)
    df['MasVnrArea'].fillna(np.round(df['MasVnrArea'].mean()),inplace=True)
    df['Electrical'].fillna(df['Electrical'].mode()[0],inplace=True)
    for feature in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCond',
                    'GarageQual', 'PoolQC', 'Fence']:
        df[feature].fillna('NA', inplace=True)

    # Remove Outliers    
    df=df[df['LotFrontage'] <= 200]
    df=df[df['LotArea'] <=180000]
    df=df[df['TotalBsmtSF'] <=3500]
    df=df[df['GrLivArea'] <=5000] 

    #Dropping unnecessary columns and feature engineering
    df.drop(['MiscFeature','Alley','MSSubClass','OverallQual','BedroomAbvGr','OverallCond','LowQualFinSF','BedroomAbvGr','KitchenAbvGr','3SsnPorch'],axis=1,inplace=True)
    df['TotBath']=df['BsmtFullBath']+df['BsmtHalfBath']+df['FullBath']+df['HalfBath']
    df.drop(['BsmtFullBath','FullBath','HalfBath','BsmtHalfBath'],axis=1,inplace=True)
    df['TotalPorchSF']=df['OpenPorchSF']+df['EnclosedPorch']+df['ScreenPorch']
    df.drop(['OpenPorchSF','EnclosedPorch','ScreenPorch','Id','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)
    df['Pool']=df['PoolArea'].apply(lambda x: 'Yes' if x >0 else 'No')
    df.drop(['PoolArea','MoSold','YrSold','MiscVal','BsmtUnfSF','GarageYrBlt','YearRemodAdd','TotRmsAbvGrd','GarageArea'],axis=1,inplace=True)
    df.drop(['LotShape','ExterQual','Condition2','Exterior1st','BsmtFinType2','Utilities','LotConfig','LandSlope','Condition1','RoofStyle','Heating'],axis=1,inplace=True)
    
    #Handling in-case missing values
    for feature in df.columns[df.isnull().sum()>0]:
        df[feature].fillna(df[feature].mode()[0], inplace=True)
    
    return df

def category_onehot(catcols, concat_df):
    df_final=concat_df.copy()
    i=0

    for f in catcols:
        temp=pd.get_dummies(concat_df[f],drop_first=True)
        concat_df.drop([f],axis=1,inplace=True)
        if i==0:
            df_final=temp.copy()
        else:
            df_final=pd.concat([df_final,temp], axis=1)
        i+=1

    df_final=pd.concat([concat_df,df_final], axis=1)
    df_final=df_final.loc[:,~df_final.columns.duplicated()]
    df_final= df_final.replace({False: 0, True: 1})

    return df_final

def train_xgb(x_train, y_train):
    regressor=xgboost.XGBRegressor(
        base_score=0.25, 
        booster='gbtree', 
        learning_rate=0.1, 
        max_depth=2, 
        n_estimators=900, 
        min_child_weight=1, 
        random_state=42
    )

    regressor.fit(x_train,y_train)
    
    filename='data/xgboost_model.pkl'
    pickle.dump(regressor,open(filename,'wb'))

def train_model(concat_df):
    df_train=concat_df.iloc[:1457,:]
    df_test=concat_df.iloc[1457:,:]
    df_test.drop(['SalePrice'], axis=1, inplace=True)
    x_train=df_train.drop(['SalePrice'], axis=1)
    y_train=df_train['SalePrice']

    train_xgb(x_train,y_train)
    

def divide_num_cat_cols(df):
    numcols=[]
    numcols_discrete=[]
    numcols_cont=[]
    catcols=[]

    for feature in df.columns:
        if(df[feature].dtype=='O'):
            catcols.append(feature)
        else:
            numcols.append(feature)
            if len(df[feature].unique())<=25:
                numcols_discrete.append(feature)
            else:
                numcols_cont.append(feature)

    return numcols,catcols,numcols_cont,numcols_discrete

def normalize(df):
    scaler = MinMaxScaler()
    df_arr=scaler.fit_transform(df)

    with open("data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return df_arr

def process_onehot_norm(df,df_test):
    _,catcols,_,_=divide_num_cat_cols(df)
    concat_df = pd.concat([df, df_test], axis=0, ignore_index=True)
    concat_df=category_onehot(catcols,concat_df)
    
    sp=pd.DataFrame(concat_df['SalePrice'])
    fitting_df=concat_df.drop(['SalePrice'],axis=1)
    concat_arr=normalize(fitting_df)
    concat_df_noSP=pd.DataFrame(concat_arr)

    concat_df=pd.concat((concat_df_noSP,sp),axis=1)

    return concat_df

