import pandas as pd
import numpy as np
import urllib.parse, re
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer

def load_and_preprocess_data():
    df = pd.read_csv("Data/properties.csv", low_memory=False)
    df = df[df["Type"]=="apartment"]
    
    df = df[['id','Price','Zip','Type','Subtype','location',
           'Surroundings type', 'Living area', 'Bedrooms','Kitchen type','Bathrooms',
           'Building condition', 'Construction year', 'Number of frontages',
           'Covered parking spaces', 'Outdoor parking spaces', 
           'Swimming pool', 'Furnished', 'How many fireplaces?',
           'Terrace','Terrace surface', 'Garden','Garden surface',
           'Primary energy consumption','Energy class','Heating type']]

    df = df.rename(columns={
        'location': 'Locality',
        'Bedrooms': 'Number of rooms',
        'Kitchen type': 'Fully equipped kitchen',
        'How many fireplaces?': 'Open fire',
        'Number of frontages': 'Number of facades'
    })

    df['Locality'] = df['Locality'].apply(urllib.parse.unquote)
    return df


def convert_and_clean(df):
    def clean_and_convert(column):
        column = column.apply(lambda x: re.sub('\D+', '', str(x)))
        column = column.replace('', np.nan)
        return column

    for col in ['Living area', 'Terrace surface', 'Garden surface', 'Primary energy consumption']:
        df[col] = clean_and_convert(df[col])
    return df


def handle_garden_terrace(df):
    for feature in ['Garden', 'Terrace']:
        conditions = [
            df[feature] == "Yes",
            df[feature].isna() & df[feature + " surface"].isna(),
            df[feature + " surface"].notna()
        ]
        values = [1, 0, 1]
        df[feature] = np.select(conditions, values)
        df.loc[(df[feature] == 0 ) & (df[feature + " surface"].isna()), feature + ' surface'] = 0
    return df


def nan_replacement(df, cols):
    for col in cols:
        df[col] = df[col].replace("Yes",1).replace("No",0).replace('', np.nan).fillna(0)
    return df


def handle_categorical_columns(df):
    kitchen_mapping = {'Not installed': 0, 'Installed': 1, 'Semi equipped': 2, 'Hyper equipped': 3, 'USA uninstalled' :0, 'USA installed': 1, 'USA semi equipped': 2, 'USA hyper equipped' :3}
    df['Kitchen values'] = df['Fully equipped kitchen'].map(kitchen_mapping).fillna(df['Fully equipped kitchen'])

    building_cond_mapping = {'To restore': 0, 'To be done up': 2, 'Just renovated': 3, 'To renovate': 1, 'Good': 3, 'As new' :4}
    df['Building Cond. values'] = df['Building condition'].map(building_cond_mapping).fillna(df['Building condition'])
    
    df = df.drop(columns=['Fully equipped kitchen', 'Building condition'])
    return df


def handle_parking(df):
    conditions = [
        (df["Covered parking spaces"].notna()) & (df["Outdoor parking spaces"].notna()),
        (df["Covered parking spaces"].isna()) & (df["Outdoor parking spaces"].isna()),
        (df["Covered parking spaces"].isna()) & (df["Outdoor parking spaces"].notna()),
        (df["Covered parking spaces"].notna()) & (df["Outdoor parking spaces"].isna())
    ]
    values = [(df["Covered parking spaces"]+df["Outdoor parking spaces"]), 0, df["Outdoor parking spaces"], df["Covered parking spaces"]]
    df['Parking'] = np.select(conditions, values)

    df = df.drop(columns=["Covered parking spaces", "Outdoor parking spaces"])
    return df


def remove_outliers(df, columns, n_std):
    for col in columns:
        mean, sd = df[col].mean(), df[col].std()
        df = df[(df[col] <= mean + n_std * sd)]
    return df


def knn_imputer(df, exclude_cols):
    other_cols = [col for col in df.columns if col not in exclude_cols]
    impute_knn = KNNImputer(n_neighbors=5)
    df[other_cols] = impute_knn.fit_transform(df[other_cols]).astype(float)
    return df


def get_province(zip_code):
    if 1000 <= zip_code <= 1299:
        return 'Brussels Capital Region'
    elif 1300 <= zip_code <= 1499:
        return 'Walloon Brabant'
    elif 1500 <= zip_code <= 1999 or 3000 <= zip_code <= 3499:
        return 'Flemish Brabant'
    elif 2000 <= zip_code <= 2999:
        return 'Antwerp'
    elif 3500 <= zip_code <= 3999:
        return 'Limburg'
    elif 4000 <= zip_code <= 4999:
        return 'Liège'
    elif 5000 <= zip_code <= 5999:
        return 'Namur'
    elif 6000 <= zip_code <= 6599 or 7000 <= zip_code <= 7999:
        return 'Hainaut'
    elif 6600 <= zip_code <= 6999:
        return 'Luxembourg'
    elif 8000 <= zip_code <= 8999:
        return 'West Flanders'
    elif 9000 <= zip_code <= 9999:
        return 'East Flanders'
    else:
        return 'Unknown'


def main():
    df = load_and_preprocess_data()
    df = convert_and_clean(df)
    df = handle_garden_terrace(df)
    df = nan_replacement(df, ['Furnished', 'Swimming pool', 'Open fire'])
    df = handle_categorical_columns(df)
    df = handle_parking(df)
    df = df.drop(df[df["Living area"].isna()].index)
    
    df = remove_outliers(df, ['Price'], 4)
    df = remove_outliers(df, ['Living area'], 3)
    df = df.drop(df[df["Number of rooms"]>6].index)
    
    df['Province'] = df['Zip'].apply(get_province)
    
    df = knn_imputer(df, ["Price", "Type of property", "Subtype of property", "Locality", "Surroundings type", "Energy class", "Heating type", "Province"])
    
    df_urbain = pd.read_csv('./Data/Urbain.csv')
    postcode_set = set(df_urbain['Postcode'])
    df['Urban_value'] = df['Zip'].apply(lambda x: 1 if x in postcode_set else 0)
    
    df["Subtype of property"] = df["Subtype of property"].replace(np.nan, "apartment")
    
    apartment = ["apartment", "ground floor", "loft", "service flat", "flat studio", "kot"]
    big_apt = ["penthouse", "triplex", "duplex"]
    
    df["Normal_apt"] = df["Subtype of property"].apply(lambda x: 1 if x in apartment else 0)
    df["Big_apt"] = df["Subtype of property"].apply(lambda x: 1 if x in big_apt else 0)
    
    df.to_csv("./Data/final_apartment.csv")

if __name__ == "__main__":
    main()

