import pandas as pd
import numpy as np
import urllib.parse
import re
from sklearn.impute import KNNImputer

def load_and_preprocess_data(file_path, property_type):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.drop(df[df["Type"]=="house group"].index)
    df = df.drop(df[df["Type"]=="apartment group"].index)
    df = df[df["Type"]==property_type]
    
    return df

def convert_and_clean(df, common_cols):
    def clean_and_convert(column):
        column = column.apply(lambda x: re.sub('\D+', '', str(x)))
        column = column.replace('', np.nan)
        return column

    for col in common_cols:
        df[col] = clean_and_convert(df[col])
    
    return df

def handle_garden_terrace(df):
    for feature in ['Garden', 'Terrace']:
        conditions = [
            df[feature]== "Yes",
            (df[feature].isna()) & (df[feature + " surface"].isna()),
            df[feature + " surface"].notna()
        ]
        values = [1, 0, 1]
        df[feature] = np.select(conditions, values)

        df.loc[(df[feature] == 0 ) & (df[feature + " surface"].isna()), feature + ' surface'] = 0
    
    return df

def nan_replacement(df, cols):
    for col in cols:
        df[col] = df[col].replace("Yes", 1).replace("No", 0).replace('', np.nan).fillna(0)
    return df

def handle_categorical_columns(df, kitchen_mapping, building_cond_mapping):
    df['Kitchen values'] = df['Fully equipped kitchen'].map(kitchen_mapping).fillna(df['Fully equipped kitchen'])
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
    values = [(df["Covered parking spaces"]+df["Outdoor parking spaces"]), 0, df["Outdoor parking spaces"],df["Covered parking spaces"]]
    df['Parking'] = np.select(conditions, values)
    
    df = df.drop(columns=["Covered parking spaces","Outdoor parking spaces"])
    
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
    
def remove_outliers(df, columns, n_std):
    for col in columns:
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean+(n_std*sd))]
    return df

def one_convert_to_nan(column):
    column = column.replace(1.0, np.nan)
    return column

def knn_imputer(df, exclude_cols):
    other_cols = [col for col in df.columns if col not in exclude_cols]
    impute_knn = KNNImputer(n_neighbors=5)
    df[other_cols] = impute_knn.fit_transform(df[other_cols]).astype(float)
    return df

def main():
    # Apartment code
    apt_df = load_and_preprocess_data("raw_data.csv", "apartment")
    common_cols = ['Living area', 'Terrace surface', 'Garden surface', 'Primary energy consumption']
    apt_df = convert_and_clean(apt_df, common_cols)
    apt_df = handle_garden_terrace(apt_df)
    apt_df = nan_replacement(apt_df, ['Furnished', 'Swimming pool', 'Open fire'])
    kitchen_mapping = {'Not installed': 0, 'Installed': 1, 'Semi equipped': 2, 'Hyper equipped': 3, 'USA uninstalled': 0,
                       'USA installed': 1, 'USA semi equipped': 2, 'USA hyper equipped': 3}
    building_cond_mapping = {'To restore': 0, 'To be done up': 2, 'Just renovated': 3, 'To renovate': 1, 'Good': 3, 'As new': 4}
    apt_df = handle_categorical_columns(apt_df, kitchen_mapping, building_cond_mapping)
    apt_df = handle_parking(apt_df)
    apt_df = apt_df.drop(apt_df[apt_df["Living area"].isna()].index)
    apt_df = apt_df.drop(apt_df[apt_df["Surface of the land"].isna()].index)
    
    apt_df['Locality'] = apt_df['Locality'].apply(urllib.parse.unquote)
    apt_df['Province'] = apt_df['Zip'].apply(get_province)
    
    apt_df = apt_df.astype({"Price": "float", "Number of rooms": "float", "Living area": "float",
                            "Terrace surface": "float", "Garden surface": "float",
                            "Number of facades": "float", "Primary energy consumption": "float"})
    
    aptdf = apt_df.copy()
    aptdf = remove_outliers(aptdf, ['Price'], 4)
    apt_df = remove_outliers(aptdf, ['Living area'], 3)
    
    apt_df.to_csv("final_apartment.csv")
    
    # House code
    house_df = load_and_preprocess_data("raw_data.csv", "house")
    common_cols = ['Living area', 'Surface of the land', 'Terrace surface', 'Garden surface', 'Primary energy consumption']
    house_df = convert_and_clean(house_df, common_cols)
    house_df = handle_garden_terrace(house_df)
    house_df = nan_replacement(house_df, ['Furnished', 'Swimming pool', 'Open fire'])
    kitchen_mapping = {'Not installed': 0, 'Installed': 1, 'Semi equipped': 2, 'Hyper equipped': 3, 'USA uninstalled': 0,
                       'USA installed': 1, 'USA semi equipped': 2, 'USA hyper equipped': 3}
    building_cond_mapping = {'To restore': 0, 'To be done up': 2, 'Just renovated': 3, 'To renovate': 1, 'Good': 3, 'As new': 4}
    house_df = handle_categorical_columns(house_df, kitchen_mapping, building_cond_mapping)
    house_df = handle_parking(house_df)
    house_df = house_df.drop(house_df[house_df["Living area"].isna()].index)
    house_df = house_df.drop(house_df[house_df["Surface of the land"].isna()].index)
    
    house_df['Locality'] = house_df['Locality'].apply(urllib.parse.unquote)
    house_df['Province'] = house_df['Zip'].apply(get_province)
    
    house_df = house_df.astype({"Price": "float", "Number of rooms": "float", "Living area": "float",
                                "Surface of the land": "float", "Terrace surface": "float", "Garden surface": "float",
                                "Number of facades": "float", "Primary energy consumption": "float"})
    
    housedf = house_df.copy()
    housedf = remove_outliers(housedf, ['Price'], 4)
    house_df = remove_outliers(housedf, ['Living area', 'Surface of the land'], 3)
    house_df['Surface of the land'] = one_convert_to_nan(house_df['Surface of the land'])
    
    house_df.to_csv("final_house.csv")

if __name__ == "__main__":
    main()
