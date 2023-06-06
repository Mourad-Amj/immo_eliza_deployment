import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib   
import os 

path_to_house_csv = '/mnt/c/Users/Moura/git/immo_eliza_deployment/Data/final_house.csv'
path_to_apartment_csv = '/mnt/c/Users/Moura/git/immo_eliza_deployment/Data/final_apartment.csv'

check_file_house = os.path.exists(path_to_house_csv)
check_file_apartment = os.path.exists(path_to_apartment_csv)

def predict() :
    if check_file_house :
      #HOUSE
      df = pd.read_csv(path_to_house_csv)
      features = list(df.drop(columns=['Unnamed: 0','Province','Zip', 'Price' , 'Locality','id', 'Primary energy consumption','Type of property','Surroundings type',
                                              'Heating type','Subtype of property', 'Energy class']).columns)

      y = df['Price']
      X = df[features].fillna(0)

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 

      # Fitting the model and making predictions
      lm = LinearRegression() 
      lm.fit(X_train, y_train) #training the algorithm

      # save the model to disk
      filename = '/mnt/c/Users/Moura/git/immo_eliza_deployment/model/LM_model_house.sav'
      joblib.dump(lm, filename)

    elif check_file_apartment:
      #Apartement
      df_aprt = pd.read_csv(path_to_apartment_csv)
      features = list(df_aprt.drop(columns=['Unnamed: 0','Province','Zip', 'Price' , 'Locality','id', 'Primary energy consumption',
                                            'Type of property','Surroundings type',
                                          'Heating type','Subtype of property', 'Energy class']).columns)

      y = df_aprt['Price']
      X = df_aprt[features].fillna(0)

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 

      # Fitting the model and making predictions
      lm = LinearRegression() 
      lm.fit(X_train, y_train) #training the algorithm

      # save the model to disk
      filename = '/mnt/c/Users/Moura/git/immo_eliza_deployment/model/LM_model_apartment.sav'
      joblib.dump(lm, filename)

  # preprocessing df from the json part of the API( app.py)
  # load the model from disk

  loaded_model = joblib.load(f'/mnt/c/Users/Moura/git/immo_eliza_deployment/model/LM_model_{df_input['type']}.sav')

  prediction = loaded_model.predict(df_input) 



      
      

return