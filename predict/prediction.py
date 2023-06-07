import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib   
import pandas as pd

kitchen_mapping = {'Not installed': 0, 'Installed': 1, 'Semi equipped': 2, 'Hyper equipped': 3, 'USA uninstalled': 0,
                       'USA installed': 1, 'USA semi equipped': 2, 'USA hyper equipped': 3}
building_cond_mapping = {'To restore': 0, 'To be done up': 2, 'Just renovated': 3, 'To renovate': 1, 'Good': 3, 'As new': 4}


path_to_house_csv = './Data/final_house.csv'
path_to_apartment_csv = './Data/final_apartment.csv'
model_path_house = './model/LM_model_house.sav'
model_path_apartment = './model/LM_model_apartment.sav'


def train_and_save_model(path_to_csv, model_path):
    df = pd.read_csv(path_to_csv)
    features = list(df.drop(columns=['Unnamed: 0', 'Province', 'Zip', 'Price', 'Locality', 'id', 
                                     'Primary energy consumption', 'Type of property', 'Surroundings type',
                                     'Heating type', 'Subtype of property', 'Energy class',
                                     'Bathrooms','Construction year','Parking']).columns)
    y = df['Price']
    X = df[features].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    joblib.dump(lm, model_path)


def predict(property_type: str, property: dict):
    if property_type.upper() == "HOUSE":
        if not os.path.exists(model_path_house):
            train_and_save_model(path_to_house_csv, model_path_house)

        loaded_model = joblib.load(model_path_house)

    elif property_type.upper() == "APARTMENT":
        if not os.path.exists(model_path_apartment):
            train_and_save_model(path_to_apartment_csv, model_path_apartment)

        loaded_model = joblib.load(model_path_apartment)
    
    df_input = pd.DataFrame({
    # 'Zip': [property['zip_code']],
    # 'Type of property': [property['property_type']],
    'Living area': [property['area']],
    'Number of rooms': [property['rooms_number']],
    'Kitchen values': [property.get('equipped_kitchen')],
    'Number of facades': [property.get('facades_number')],
    'Swimming pool': [property.get('swimming_pool')],
    'Furnished': [property.get('furnished')],
    'Open fire': [property.get('open_fire')],
    'Terrace': [property.get('terrace')],
    'Terrace surface': [property.get('terrace_area')],
    'Garden': [property.get('garden')],
    'Garden surface': [property.get('garden_area')],
    'Surface of the land': [property.get('land_area')],
    'Building Cond. values': [property.get('building_state')]
}).fillna(0)
    
  # Re-order the columns in prediction data to match the order in training data
    df_input = df_input[loaded_model.feature_names_in_]
    df_input['Kitchen values'] = df_input['Kitchen values'].map(kitchen_mapping)
    df_input['Building Cond. values'] = df_input['Building Cond. values'].map(building_cond_mapping)

    prediction = loaded_model.predict(df_input)

    return prediction.tolist()
