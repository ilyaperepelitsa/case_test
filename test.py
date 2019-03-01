import pandas as pd
import numpy as np
from datetime import datetime
from geopy.distance import vincenty
from geopy.distance import geodesic
from geopy import Point

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
import math

pd.set_option('display.float_format', lambda x: '%.3f' % x)


facts = pd.read_excel("01_Facts.xlsx", header = None)
event_type_data = pd.read_excel("04_event_type.xlsx")
event_type_data = event_type_data.rename(columns = {"Номер":"event_type", "Обозначение":"event_symbol", "Описание":"event_description"})
event_type_data = event_type_data.astype({"event_type" : int})
# event_type.to_dict("records")
test = pd.read_csv("02_Data_test.csv", sep = ";", dtype = {"imei" : "object",
                                                            "lac" : "object",
                                                            "cid" : "object",
                                                            "msisdn" : "object",
                                                            "event_type" : int})
test['tstamp'] = pd.to_datetime(test['tstamp'], unit='ms')
# test

# test.groupby(["cid"])["max_dist"].describe()
devices = pd.read_csv("03_устройства.csv", sep = "\"*,\"*")
devices = devices.applymap(lambda x: x.replace("\"", ""))

devices.columns = devices.columns.str.replace("\"", "")

test["tac"] = test["imei"].str.extract("^(\d{8})")
test = test.merge(devices, on = "tac", how = "left")
test = test.merge(event_type_data, on = "event_type")

test = test.sort_values(["msisdn", "tstamp"], ascending = False)

test["lat_prev"] = test.groupby(["msisdn"])["lat"].shift(-1)
test["long_prev"] = test.groupby(["msisdn"])["long"].shift(-1)

test["previous_timestamp"] = test.groupby(["msisdn"])["tstamp"].shift(-1)

test.loc[~test["lat_prev"].isnull(),"previous_status_distance"] = test.loc[~test["lat_prev"].isnull(),:].\
                                        apply(lambda x: geodesic((x["lat"],
                                                                    x["long"]),
                                                                (x["lat_prev"],
                                                                x["long_prev"])).meters, axis = 1)


test["date"] = test["tstamp"].dt.date

def circle_segment(data):
    origin = Point(data["lat"], data["long"])
    destination = geodesic(kilometers=data["max_dist"]/1000).destination(origin, data["start_angle"])
    lat2, lon2 = destination.latitude, destination.longitude
    # print(type(lat2))
    return lat2, lon2
    # print(pd.concat(lat2, lon2), axis = 0)
    # return pd.DataFrame(pd.Series(lat2), pd.Series(lon2))

def sector_centroid(data):
    origin = Point(data["lat"], data["long"])
    angle 
    destination = geodesic(kilometers=data["max_dist"]/1000).destination(origin, data["start_angle"])
    lat2, lon2 = destination.latitude, destination.longitude
    # print(type(lat2))
    return lat2, lon2
    # print(pd.concat(lat2, lon2), axis = 0)
    # return pd.DataFrame(pd.Series(lat2), pd.Series(lon2))



# test["segment_lat"], test["segment_lon"] = test.apply(circle_segment, axis = 1)


test["segment_lat"], test["segment_lon"] = test.apply(circle_segment, axis = 1).apply(lambda x: x[0]), test.apply(circle_segment, axis = 1).apply(lambda x: x[1])

test["plot_radius"] = test.apply(lambda x: math.hypot(x["segment_lat"] - x["lat"], x["segment_lon"] - x["long"]), axis = 1)
test["station_angle"] = test.end_angle - test.start_angle
# test["station_angle"][test["station_angle"] < 0] += 360
test["station_angle"][test["station_angle"] < 0] = test["station_angle"][test["station_angle"] < 0] + 360

test["sector_centroid_shift"] = (2*test["max_dist"] * np.sin(test["station_angle"])) / 3*np.sin(test["station_angle"])
# plt.hist(np.log((2*test["max_dist"] * np.sin(test["station_angle"])) / 3*np.sin(test["station_angle"])))


# test["station_angle"].columns

test.columns

pd.concat([test.end_angle - test.start_angle, test.start_angle, test.end_angle], axis =  1)

a = pd.Series(test.end_angle - test.start_angle)
a[a<0] += 360
# Here's a simpler workaround. Use the hatch argument in your mpatches.Arc command. If you repeat symbols with the hatch argument it increases the density of the patterning. I find that if you use 6 dashes, '-', or 6 dots, '.' (others probably also work), then it solidly fills in the arc as desired. When I run this



# plt.axes()
plt.figure(figsize=(20,10))
for x, i in test.drop_duplicates("cid").loc[test.lac == test.lac.unique()[3],:].iterrows():
    # print(i["cid"])
    x = i["long"]
    y = i["lat"]
    # angle = i["end_angle"] - i["start_angle"]
    start_angle = i["start_angle"]
    end_angle = i["end_angle"]
    radius = i["plot_radius"]


# pac = mpatches.Wedge(center = [x, y], r = radius, theta1=start_angle + 90, theta2=end_angle)
# pac = mpatches.Wedge(center = [x, y], r = radius, theta1=0, theta2=15)
# pac_2 = mpatches.Wedge(center = [x + 0.1, y + 0.1], r = radius, theta1=0, theta2=35)
    pac_2 = mpatches.Wedge(center = [x, y], r = radius, theta1=start_angle, theta2=end_angle, alpha = 0.1)
# plt.gca().add_patch(pac)
    plt.gca().add_patch(pac_2)
    pac_2.set_color('cyan')
plt.axis('equal')
plt.show()

math.pi



from matplotlib.patches import Arc as arc


fig_width, fig_height = 3.30, 3.30
fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], aspect='equal')



fig = plt.figure()

ax = plt.axes(xlim = (-10,10), ylim = (-10,10))



print()

# test["max_dist"]
# a1, a2 = test.head().apply(circle_segment, axis = 1).apply(lambda x: x[0]), test.head().apply(circle_segment, axis = 1).apply(lambda x: x[0])


# a1
# test.head()
#
# test.apply(lambda x: (x["lat"], x["long"]))
# test.apply(lambda x: print(x["lat"]))
#
#
# event_type_data.event_type.values
# test.event_type.values
#
#
# .astype({"event_type" : "object"})
#
# pd.Series(test.event_type.values).isin(event_type_data.event_type.values).any()
#
#
# test.merge(event_type)
#
# test.
# event_type.info()
# test.info()
#
# test.imei.isnull().sum()
# test.msisdn.isnull().sum()
# test
#
#
#
#
#
#
#
# test.describe()
# test.info()
#
# test["msisdn"].unique().shape[0]
# test.sort_values(["msisdn", "tstamp"])
