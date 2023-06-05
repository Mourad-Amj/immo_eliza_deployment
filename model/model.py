import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, datafile):
        self.datafile = datafile
        self.model = None
        self.selector = SelectKBest(score_func=f_regression, k=18)

    def train_model(self):
        df = pd.read_csv(self.datafile)
        features = list(df.drop(columns=['Zip', 'Price', 'id', 'Primary energy consumption',
                                         'Subtype of property', 'Energy class', 'Province']).columns)
        X = df[features]
        y = df['Price']

        X_selected = self.selector.fit_transform(X, y)
        self.model = LinearRegression()
        self.model.fit(X_selected, y)

    def predict_price(self, property_data):
        if self.model is None:
            self.train_model()
        
        selected_features = self.selector.transform(property_data)
        prediction = self.model.predict(selected_features)
        return prediction[0]

def load_model(property_type: str):
    if property_type == "APARTMENT":
        datafile = "./Data/final_apartment.csv"
    elif property_type == "HOUSE":
        datafile = "./Data/final_house.csv"
    else:
        raise ValueError("It should be either 'APARTMENT' or 'HOUSE'.")
    
    model = Model(datafile)
    return model
