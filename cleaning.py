import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def preprocessing(read_csv):
    df = pd.read_csv(read_csv)  # reading the csv file

    # df = df.drop(columns=['Unnamed: 0'])  # dropping the unexpected column

    if df.isna().sum().sum() > 0:  # checking the null values' presence
        df = df.apply(
            lambda cols: cols.fillna(cols.mean()) if np.issubdtype(cols.dtype, np.number)
            else cols.fillna(cols.mode()[0])
        )  # filling the null values

    if df.duplicated().sum() > 0:  # cheacking the duplicate values' presence
        df = df.drop_duplicates()  # clipping off the duplicates

    # ['murder' = 0, 'rape' = 1, 'assault' = 2,  'bodyfound' = 3,  'kidnap' = 4, 'robbery' = 5]

    df['crime'] = df['crime'].apply(
        lambda x: 0 if x == 'murder' else 1 if x == 'rape' else 2 if x == 'assault'
        else 3 if x == 'bodyfound' else 4 if x == 'kidnap' else 5
    )  # encoding the categorical label

    encoder = OrdinalEncoder()  # defining the encoder
    df_cats = df.select_dtypes(include=['object']).columns.tolist()  # selecting the categorical features

    if len(df_cats) > 0:
        df[df_cats] = encoder.fit_transform(df[df_cats])  # encoding the categorical features
        joblib.dump(encoder, 'encoder.pkl')  # dumping the encoder

    return df


cleaned_df = preprocessing(r"C:\Users\JUNAID AHMED\Downloads\Crime\crime_dataset.csv")  #
joblib.dump(cleaned_df, 'df.pkl')  # dumping the cleanded dataframe