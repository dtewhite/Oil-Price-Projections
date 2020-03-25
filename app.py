import pandas as pd
import json
import requests
from functools import reduce
from keys import fred_key, eia_key, twitter_api_key, twitter_api_secret_key, twitter_access_token, twitter_access_secret_token
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from dateutil.relativedelta import relativedelta
import tweepy

# EIA API Calls
production = requests.get("https://api.eia.gov/series/?api_key="+ eia_key +"&series_id=PET.WCRFPUS2.W")
imports = requests.get("http://api.eia.gov/series/?api_key="+ eia_key +"&series_id=PET.WCRIMUS2.W")
supply = requests.get("http://api.eia.gov/series/?api_key="+ eia_key +"&series_id=PET.WRPUPUS2.W")

# FRED API Calls
# The WTI returns price for the week ending Friday. This is the same release dat as the eia data
wti = requests.get("https://api.stlouisfed.org/fred/series/observations?series_id=WCOILWTICO&frequency=wef&api_key="+ 
                   fred_key +"&file_type=json")

# These indicators are only monthly
cpi = requests.get("https://api.stlouisfed.org/fred/series/observations?series_id=CPILFESL&frequency=m&api_key="+ 
                  fred_key +"&file_type=json")
unemployment = requests.get("https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&frequency=m&api_key="+ 
                  fred_key +"&file_type=json")
personal_consumption_expenditure = requests.get("https://api.stlouisfed.org/fred/series/observations?series_id=PCE&frequency=m&api_key="+ 
                  fred_key +"&file_type=json")

# Convert responses to JSONs
production_json = production.json()
imports_json = imports.json()
supply_json = supply.json()
wti_json = wti.json()
cpi_json = cpi.json()
unemp_json = unemployment.json()
pce_json = personal_consumption_expenditure.json()

# Extract EIA Data
production_series = production_json['series'][0]
imports_series = imports_json['series'][0]
supply_series = supply_json['series'][0]
production_df = pd.DataFrame(production_series['data'])
imports_df = pd.DataFrame(imports_series['data'])
supply_df = pd.DataFrame(supply_series['data'])

# Extract FRED Data
wti_observations = wti_json['observations']
Date = []
Value = []
for observation in wti_observations:
    Date.append(observation['date'])
    Value.append(observation['value'])
wti_df = pd.DataFrame(list(zip(Date,Value)))

cpi_observations = cpi_json['observations']
Date = []
Value = []
for observation in cpi_observations:
    Date.append(observation['date'])
    Value.append(observation['value'])
cpi_df = pd.DataFrame(list(zip(Date,Value)))

unemp_observations = unemp_json['observations']
Date = []
Value = []
for observation in unemp_observations:
    Date.append(observation['date'])
    Value.append(observation['value'])
unemp_df = pd.DataFrame(list(zip(Date,Value)))

pce_observations = pce_json['observations']
Date = []
Value = []
for observation in pce_observations:
    Date.append(observation['date'])
    Value.append(observation['value'])
pce_df = pd.DataFrame(list(zip(Date,Value)))

# Converting to datetime 
production_df.iloc[:,0] = pd.to_datetime(production_df.iloc[:,0],format='%Y%m%d', errors='raise')
imports_df.iloc[:,0] = pd.to_datetime(production_df.iloc[:,0],format='%Y%m%d', errors='raise')
supply_df.iloc[:,0] = pd.to_datetime(production_df.iloc[:,0],format='%Y%m%d', errors='raise')
wti_df.iloc[:,0] = pd.to_datetime(wti_df.iloc[:,0],format='%Y-%m-%d', errors='raise')
cpi_df.iloc[:,0] = pd.to_datetime(cpi_df.iloc[:,0],format='%Y-%m-%d', errors='raise')
unemp_df.iloc[:,0] = pd.to_datetime(unemp_df.iloc[:,0],format='%Y-%m-%d', errors='raise')
pce_df.iloc[:,0] = pd.to_datetime(pce_df.iloc[:,0],format='%Y-%m-%d', errors='raise')

# Rename columns
production_df = production_df.rename(columns = {0:'Date', 1:'Production (thousand barrels per day)'}).sort_values(by='Date').reset_index(drop=True)
imports_df = imports_df.rename(columns = {0:'Date', 1:'Imports (thousand barrels per day)'}).sort_values(by='Date').reset_index(drop=True)
supply_df = supply_df.rename(columns = {0:'Date', 1:'Supply (thousand barrels per day)'}).sort_values(by='Date').reset_index(drop=True)
wti_df = wti_df.rename(columns = {0:'Date', 1:'Price of Barrel (usd)'}).sort_values(by='Date')
cpi_df = cpi_df.rename(columns = {0:'Date', 1:'Core CPI (index 1982-1984=100)'}).sort_values(by='Date')
unemp_df = unemp_df.rename(columns = {0:'Date', 1:'Unemployment (%)'}).sort_values(by='Date')
pce_df = pce_df.rename(columns = {0:'Date', 1:'Personal Consumption Expenditure (billions of usd)'}).sort_values(by='Date')

# Merge EIA data and WTI
data_frames = [supply_df,imports_df,production_df,wti_df]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'], how='left'), data_frames)

# Merge FRED data
data_frames = [cpi_df, pce_df, unemp_df]
indicators_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'], how='left'), data_frames)

# Creating new datetime columns to merge on
indicators_merged['Merge Date']= indicators_merged['Date'].dt.strftime('%m-%Y')
df_merged['Merge Date'] = df_merged['Date'].dt.strftime('%m-%Y')

# Merging and cleaning columns
final_df = pd.merge(df_merged, indicators_merged, how='left', on='Merge Date').fillna(method='ffill')
final_df = final_df[['Date_x', 'Merge Date', 'Personal Consumption Expenditure (billions of usd)','Core CPI (index 1982-1984=100)',
          'Unemployment (%)', 'Supply (thousand barrels per day)', 'Imports (thousand barrels per day)', 'Production (thousand barrels per day)', 
          'Price of Barrel (usd)']]
final_df = final_df.rename(columns = {"Date_x":"Date", "Merge Date":"Month"})
final_df.iloc[-1,:].values

# Staggering the price of WTI. This is so our independent variables are actually trained to predict next week's price.

PriceOfBarrel = final_df["Price of Barrel (usd)"]
StaggeredList=[]
count=1
while(count<len(PriceOfBarrel)):
    StaggeredList.append(PriceOfBarrel[count])
    count +=1
StaggeredList.append(0)
modeling_df = final_df.copy()
modeling_df ["Staggered Price of Barrel"] = StaggeredList
modeling_df = modeling_df.iloc[:-1]

modeling_df['Personal Consumption Expenditure (billions of usd)'] = pd.to_numeric(modeling_df['Personal Consumption Expenditure (billions of usd)'],errors='coerce')
modeling_df['Core CPI (index 1982-1984=100)'] = pd.to_numeric(modeling_df['Core CPI (index 1982-1984=100)'],errors='coerce')
modeling_df['Unemployment (%)'] = pd.to_numeric(modeling_df['Unemployment (%)'],errors='coerce')
modeling_df['Price of Barrel (usd)'] = pd.to_numeric(modeling_df['Price of Barrel (usd)'],errors='coerce')
modeling_df['Staggered Price of Barrel'] = pd.to_numeric(modeling_df['Staggered Price of Barrel'],errors='coerce')

modeling_df.to_csv("Data.csv")

# Establishing dependent and independent variables
X = modeling_df[["Personal Consumption Expenditure (billions of usd)",
              "Unemployment (%)",
              "Supply (thousand barrels per day)",
              "Imports (thousand barrels per day)",
              "Production (thousand barrels per day)"]]
y = modeling_df["Staggered Price of Barrel"].values.reshape(-1, 1)

# Splitting training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scaling data

X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Random Forest Model

rfm = RandomForestRegressor(n_estimators=1000)

rfm.fit(X_train_scaled, y_train_scaled)

training_score = rfm.score(X_train_scaled, y_train_scaled)
testing_score = rfm.score(X_test_scaled, y_test_scaled)

# Calculating prediction
most_recent_X = final_df.iloc[-1,:]
most_recent_X = X_scaler.transform([most_recent_X[X.columns].values])
price_next_week = rfm.predict(most_recent_X)
price_next_week = y_scaler.inverse_transform(price_next_week)[0].round(2)

# Calculating next Friday's date
most_recent_date = (final_df.iloc[-1,0]).date()
next_week_date = most_recent_date + relativedelta(weeks=+1)

# Pulling this past Friday's price
most_recent_price = str(final_df.iloc[-1,8])

# Connecting to Twitter API
auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret_key)
auth.set_access_token(twitter_access_token, twitter_access_secret_token)

api = tweepy.API(auth)

# Confirming Authentication is alright
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

# Posting Tweet of new data
api.update_status("The price of oil (WTI) on Friday, " + str(most_recent_date) + ", was $" + 
        str(most_recent_price) + ". Our model predicts that the price of oil on Friday, " + 
        str(next_week_date) + ", will be $" + str(price_next_week) + ".")