

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.parse, re
import matplotlib

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.impute import KNNImputer



df = pd.read_csv("raw_data.csv", low_memory=False)
df = df.drop(df[df["Type"]=="house group"].index)
df = df.drop(df[df["Type"]=="apartment group"].index)

df = df[df["Type"]=="house"]




df = df[['id','Price','Zip','Type','Subtype','location',
       'Surroundings type',
       'Living area','Surface of the plot',
       'Bedrooms','Kitchen type','Bathrooms',
       'Building condition',
       'Construction year', 
       'Number of frontages',
       'Covered parking spaces', 'Outdoor parking spaces', 
       'Swimming pool',
       'Furnished',
       'How many fireplaces?',
       'Terrace','Terrace surface',
       'Garden','Garden surface',
       'Primary energy consumption','Energy class','Heating type'
       ]]



df = df.rename(columns={
    'location' :'Locality',
    'Transaction Type' : 'Type of sale',
    'Type' :'Type of property',
    'Subtype' : 'Subtype of property',
    'Number of frontages': 'Number of facades',
    'Bedrooms':'Number of rooms',
    'Surface of the plot' :'Surface of the land',
    'Kitchen type' : 'Fully equipped kitchen',
    'How many fireplaces?' : 'Open fire',
})

df['Locality'] = df['Locality'].apply(urllib.parse.unquote)

def clean_and_convert(column):
    column = column.apply(lambda x: re.sub('\D+', '', str(x)))
    column = column.replace('', np.nan)
    return column

df['Living area'] = clean_and_convert(df['Living area'])
df['Terrace surface'] = clean_and_convert(df['Terrace surface'])
df['Garden surface'] = clean_and_convert(df['Garden surface'])
df['Surface of the land'] = clean_and_convert(df['Surface of the land'])
df['Primary energy consumption'] = clean_and_convert(df['Primary energy consumption'])
# GARDEN AND TERRACE
conditions = [
    (df['Garden']== "Yes"),
    (df["Garden"].isna()) & (df["Garden surface"].isna()),
    (df["Garden surface"].notna())
    ]
values = [1, 0, 1]
df['Garden'] = np.select(conditions, values)

df.loc[(df["Garden"] == 0 ) & (df["Garden surface"].isna()), 'Garden surface'] = 0

conditions = [
    (df['Terrace']== "Yes"),
    (df["Terrace"].isna()) & (df["Terrace surface"].isna()),
    (df["Terrace surface"].notna())
    ]
values = [1, 0, 1]
df['Terrace'] = np.select(conditions, values)

df.loc[(df["Terrace"] == 0 ) & (df["Terrace surface"].isna()), 'Terrace surface'] = 0

def nan_replacement(column):
    column = column.replace("Yes",1)
    column = column.replace("No",0)
    column = column.replace('', np.nan).fillna(0)
    return column

df['Furnished'] = nan_replacement(df['Furnished'])
df['Swimming pool'] = nan_replacement(df['Swimming pool'])
df['Open fire'] = nan_replacement(df['Open fire'])

# Mapping dictionary for replacing values in the "kitchen" column
kitchen_mapping = {
    # np.nan: -1,
    'Not installed': 0,
    'Installed': 1,
    'Semi equipped': 2,
    'Hyper equipped': 3,
    'USA uninstalled' :0,
    'USA installed': 1,
    'USA semi equipped': 2,
    'USA hyper equipped' :3
}
# Replace values in the "Kitchen type" column with corresponding numbers and create a new column called "Kitchen values"
df['Kitchen values'] = df['Fully equipped kitchen'].map(kitchen_mapping).fillna(df['Fully equipped kitchen'])

building_cond_mapping = {
    # np.nan: -1,
    'To restore': 0,
    'To be done up': 2,
    'Just renovated': 3,
    'To renovate': 1,
    'Good': 3,
    'As new' :4
}

df['Building Cond. values'] = df['Building condition'].map(building_cond_mapping).fillna(df['Building condition'])

df = df.loc[:, ~df.columns.isin(['Fully equipped kitchen','Building condition'])]



df = df.drop(df[df["Living area"].isna()].index)
df = df.drop(df[df["Surface of the land"].isna()].index)



conditions = [
    (df["Covered parking spaces"].notna()) & (df["Outdoor parking spaces"].notna()),
    (df["Covered parking spaces"].isna()) & (df["Outdoor parking spaces"].isna()),
    (df["Covered parking spaces"].isna()) & (df["Outdoor parking spaces"].notna()),
    (df["Covered parking spaces"].notna()) & (df["Outdoor parking spaces"].isna())
    ]
values = [(df["Covered parking spaces"]+df["Outdoor parking spaces"]), 0, df["Outdoor parking spaces"],df["Covered parking spaces"]]
df['Parking'] = np.select(conditions, values)

df = df.loc[:, ~df.columns.isin(["Covered parking spaces","Outdoor parking spaces"])]



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
        
df['Province'] = df['Zip'].apply(get_province)

df = df.astype({"Price":"float",
                "Number of rooms":"float",
                "Living area":"float",
                "Terrace surface":"float",
                "Garden surface":"float",
                "Surface of the land":"float",
                "Number of facades":"float",
                "Primary energy consumption":"float"})

housedf = df.copy()

print("House DataFrame shape (before): ",housedf.shape)
print("House data min (with outliers): ",housedf['Price'].min())
print("House data max (with outliers): ",housedf['Price'].max())

# Remove outliers
def remove_outliers(df, columns, n_std) -> pd.DataFrame:
    for col in columns:
        print('Working on column: {}'.format(col))
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean+(n_std*sd))]
    return df

new_housedf = remove_outliers(housedf, ['Price'], 4)



def one_convert_to_nan(column):
    column = column.replace(1.0, np.nan)
    return column
house_df = remove_outliers(new_housedf, ['Living area','Surface of the land'], 3)

house_df['Surface of the land'] = one_convert_to_nan(house_df['Surface of the land'])



knn_df = house_df.loc[:, ~house_df.columns.isin(["Price","Type of property","Subtype of property","Locality","Surroundings type","Energy class","Heating type","Province"])]
other = ['id', 'Zip', 'Living area', 'Surface of the land', 'Number of rooms',
       'Bathrooms', 'Construction year', 'Number of facades', 'Swimming pool',
       'Furnished', 'Open fire', 'Terrace', 'Terrace surface', 'Garden',
       'Garden surface', "Primary energy consumption",'Kitchen values', 'Building Cond. values', 'Parking']

impute_knn = KNNImputer(n_neighbors=5)

knn_df = impute_knn.fit_transform(knn_df).astype(float)

#Creating dfs with missing values filled in 
imputed_houses = pd.DataFrame(knn_df, columns= other)

#Creating dfs with prices
new_houses = house_df[['id',"Price","Type of property","Subtype of property","Locality","Surroundings type","Energy class","Heating type","Province"]]

#Merging dfs (with prices and without prices (with other values filled in))
complete_houses = pd.merge(new_houses, imputed_houses,on='id')



df_urbain = pd.read_csv('Urbain.csv')
postcode_set = set(df_urbain['Postcode'])
complete_houses['Urban_value'] = complete_houses['Zip'].apply(lambda x: 1 if x in postcode_set else 0)



mansion = ["manor house","mansion","castle","exceptional property"]
house = ["house","villa","bungalow","chalet","country cottage","farmhouse","mixed use building","town house"]
other = ["apartment block","other property"]

complete_houses["Mansion"] = complete_houses["Subtype of property"].apply(lambda x: 1 if x in mansion else 0)
complete_houses["House_villa"] = complete_houses["Subtype of property"].apply(lambda x: 1 if x in house else 0)
complete_houses["Other_house"] = complete_houses["Subtype of property"].apply(lambda x: 1 if x in other else 0)



complete_houses.to_csv("final_house.csv")



