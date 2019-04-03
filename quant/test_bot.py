import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/ilyaperepelitsa/Downloads/test_data.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data["date"] = data['timestamp'].dt.date
data_one_day = data.loc[data.date == data.date.unique()[0],:]
#
# data_one_day['timestamp'].dt.time
#
# data_one_day.info()

#
plt.figure(figsize=(26,7))
plt.hist(data.loc[data.date == data.date.unique()[1],"timestamp"], bins = 500)
plt.show()
#
plt.figure(figsize=(26,7))
plt.hist(data.loc[data.date == data.date.unique()[2],"timestamp"], bins = 500)
plt.show()

plt.figure(figsize=(26,7))
plt.hist(data.loc[data.date == data.date.unique()[3],"timestamp"], bins = 200)
plt.show()

plt.figure(figsize=(26,7))
plt.hist(data.loc[data.date == data.date.unique()[4],"timestamp"], bins = 200)
plt.show()


# data_one_day

data = data.sort_values("timestamp")
data['prev_timestamp'] = data.groupby(["date"])["timestamp"].shift(-1)
data.sort_values("timestamp")

data["delta"] = (data.loc[:,"prev_timestamp"] - data.loc[:,"timestamp"]).dt.microseconds


# plt.hist(data["delta"]*100)
# plt.hist(data["delta"].dropna().dt.microseconds)
#
# from sklearn.preprocessing import StandardScaler
# # data.date == data.date.unique()[0]
# scaler = StandardScaler()

# scaler.fit_transform(data["delta"].dropna().dt.microseconds.reshape(-1, 1))

# data["delta_norm"] = (data["delta"].dropna().dt.microseconds - data["delta"].dropna().dt.microseconds.mean())/data["delta"].dropna().dt.microseconds.std()

data.dropna()

import fbprophet
data = data.rename(columns={'timestamp': 'ds', 'delta': 'y'})
# Put market cap in billions
# gm['y'] = gm['y'] / 1e9
# Make the prophet model and fit on the data
gm_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)
gm_prophet.fit(data)


gm_prophet.changepoints[:10]
