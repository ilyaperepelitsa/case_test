import pandas as pd
import numpy as np
from datetime import datetime
from geopy.distance import vincenty
from geopy.distance import geodesic
from geopy import Point
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
import math

import os
pd.set_option('display.float_format', lambda x: '%.7f' % x)


palette_pastel = ['#bcc19c','#c19c9e','#c19cbc','#c1a09c','#c19cac','#9c9dc1','#9cb0c1',
            '#c1b19c','#b29cc1','#9cc1bd','#9cc1a2']

from itertools import cycle
from cycler import cycler

color_cycler = cycle(plt.rcParams["axes.prop_cycle"])
cmap = lambda: next(color_cycler)["color"]


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
    angle = data["end_angle"] - data["start_angle"]
    angle_mid = data["start_angle"] + angle / 2
    # if angle_mid > 360:
    #     angle_mid = angle_mid - 180
# if data["end_angle"] - data["start_angle"] >= 180:
#     destination = geodesic(kilometers=-data["sector_centroid_shift"]/1000).destination(origin, angle_mid)
# else:
    destination = geodesic(kilometers=data["sector_centroid_shift"]/1000).destination(origin, angle_mid)
    lat2, lon2 = destination.latitude, destination.longitude

    return lat2, lon2


# def sector_angle_mid(data):
#     angle_mid = (data["end_angle"] + data["start_angle"]) / 2
#     if angle_mid > 360:
#         angle_mid = angle_mid - 180
#     # print(type(lat2))
#     return angle_mid
#
# def get_cmap(n, name='hsv'):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     return plt.cm.get_cmap(name, n)

# test["segment_lat"], test["segment_lon"] = test.apply(circle_segment, axis = 1)

test["segment_lat"], test["segment_lon"] = test.apply(circle_segment, axis = 1).apply(lambda x: x[0]), test.apply(circle_segment, axis = 1).apply(lambda x: x[1])

test["plot_radius"] = test.apply(lambda x: math.hypot(x["segment_lon"] - x["long"], x["segment_lat"] - x["lat"]), axis = 1)
# test["plot_radius_2"] = test.apply(lambda x: math.sqrt((x["segment_lon"] - x["long"])**2 + (x["segment_lat"] - x["lat"])**2), axis = 1)

test["station_angle"] = test.end_angle - test.start_angle
# test["station_angle"][test["station_angle"] < 0] += 360
test["station_angle"][test["station_angle"] < 0] = test["station_angle"][test["station_angle"] < 0] + 360

test["sector_centroid_shift"] = test.apply(lambda x: (2*x["max_dist"] * np.sin(math.radians(x["station_angle"]))) / (3*math.radians(x["station_angle"])),axis = 1)
test["sector_centroid_lat"], test["sector_centroid_lon"] = test.apply(sector_centroid, axis = 1).apply(lambda x: x[0]), test.apply(sector_centroid, axis = 1).apply(lambda x: x[1])

#
# test["sector_centroid_shift"]
# np.sin(math.degrees(45))
#
# # pd.concat([test["plot_radius"], test["plot_radius_2"]], axis = 1)
# pd.concat([test["sector_centroid_shift"], test["max_dist"]], axis = 1)
#


# pd.concat([test["start_angle"], test.apply(sector_angle_mid, axis = 1), test["end_angle"]], axis = 1)
# plt.hist(np.log((2*test["max_dist"] * np.sin(test["station_angle"])) / 3*np.sin(test["station_angle"])))
# pd.concat([test["start_angle"], test["station_angle"], test["end_angle"]], axis = 1)

#
# test[["sector_centroid_lat","sector_centroid_lon"]]
#
#
# test.head().apply(lambda x: x["max_dist"]/1000, axis = 1)
#
#
# geodesic(kilometers=data["max_dist"]/1000)
#
#
# 350 + 100
# (350 + 180)/2 -360/2
# # test["station_angle"].columns
#
# test.columns
# (test.end_angle > test.start_angle).sum()
#
# pd.concat([test.end_angle - test.start_angle, test.start_angle, test.end_angle], axis =  1)
#
# a = pd.Series(test.end_angle - test.start_angle)
# a[a<0] += 360
# # Here's a simpler workaround. Use the hatch argument in your mpatches.Arc command. If you repeat symbols with the hatch argument it increases the density of the patterning. I find that if you use 6 dashes, '-', or 6 dots, '.' (others probably also work), then it solidly fills in the arc as desired. When I run this
#
#

# plt.axes()
plt.figure(figsize=(20,10))
# cmap = get_cmap(test.drop_duplicates("cid").shape[0])
# new_cmap = rand_cmap(100, type='bright', first_color_black=True, last_color_black=False, verbose=True)
# len(new_cmap)
for ix, i in test.drop_duplicates("cid").iterrows():
# i = test.loc[test.index[900],:]
    # print(i["cid"])
    x = i["long"]
    y = i["lat"]
    start_angle = i["start_angle"]
    end_angle = i["end_angle"]
    radius = i["plot_radius"]
    pac_2 = mpatches.Wedge(center = [x, y], r = radius, theta2=-270 - start_angle, theta1=-270-end_angle, alpha = 0.3)
    plt.gca().add_patch(pac_2)

plt.axis('equal')
# plt.show()
plt.savefig("all_towers.png")





# plt.axes()
plt.figure(figsize=(20,10))
# cmap = get_cmap(test.drop_duplicates("cid").shape[0])
# new_cmap = rand_cmap(100, type='bright', first_color_black=True, last_color_black=False, verbose=True)
# len(new_cmap)
for ix, i in test.drop_duplicates("cid").loc[test.drop_duplicates("cid").index[500:505],:].iterrows():
# i = test.loc[test.index[900],:]
    # print(i["cid"])
    x = i["long"]
    y = i["lat"]
    start_angle = i["start_angle"]
    end_angle = i["end_angle"]
    radius = i["plot_radius"]
    pac_2 = mpatches.Wedge(center = [x, y], r = radius, theta2=-270 - start_angle, theta1=-270-end_angle, alpha = 0.3)
    plt.gca().add_patch(pac_2)
    plt.scatter(i["sector_centroid_lon"], i["sector_centroid_lat"])
plt.axis('equal')
# plt.show()
plt.savefig("five_towers_centroids.png")




# plt.axes()
plt.figure(figsize=(20,10))
for ix, i in test.drop_duplicates("cid").loc[test.drop_duplicates("cid").index[700:750],:].iterrows():
    x = i["long"]
    y = i["lat"]
    start_angle = i["start_angle"]
    end_angle = i["end_angle"]
    radius = i["plot_radius"]
    pac_2 = mpatches.Wedge(center = [x, y], r = radius, theta2=-270 - start_angle, theta1=-270-end_angle, alpha = 0.3)
    plt.gca().add_patch(pac_2)
    plt.scatter(i["sector_centroid_lon"], i["sector_centroid_lat"])
plt.axis('equal')
# plt.show()
plt.savefig("fifty_towers_centroids.png")


list_stuff = test.loc[test['msisdn'].isin(test['msisdn'].unique()[0:10]),'msisdn'].unique()
# list_stuff


stack_events = pd.DataFrame()
for index, combo in enumerate(list(list(i) for i in combinations(test['msisdn'].unique(), 2))[0:10]):
    # print(test.loc[test['msisdn']==combo[0],:].shape, test.loc[test['msisdn'] == combo[1],:].shape)
    event_frame = test.loc[test['msisdn'].isin(combo),:].copy()
    event_frame.loc[:,"combo_id"] = index
    # print(event_frame.head())
    # event_frame.to_csv(os.path.join("combos", str(index) + ".csv"))

    event_frame = event_frame.drop(['lac', 'imei', 'max_dist', 'event_type',
                'lat_prev', 'long_prev', 'previous_timestamp', 'previous_status_distance',
                'segment_lat', 'segment_lon',
                'station_angle', 'sector_centroid_shift'], axis = 1).\
                sort_values(['combo_id', 'tstamp'], ascending = False)

    event_frame["msisdn_lag"] = event_frame.groupby(["combo_id"])["msisdn"].shift(-1)
    event_frame.loc[event_frame["msisdn_lag"] != event_frame["msisdn"],:]

    event_frame["tstamp_lag"] = event_frame.groupby(["combo_id"])["tstamp"].shift(-1)
    event_frame["sector_centroid_lat_lag"] = event_frame.groupby(["combo_id"])["sector_centroid_lat"].shift(-1)
    event_frame["sector_centroid_lon_lag"] = event_frame.groupby(["combo_id"])["sector_centroid_lon"].shift(-1)

    stack_events = pd.concat([stack_events, event_frame], axis = 0)

from matplotlib.collections import LineCollection

plt.figure(figsize=(20,10))
id_id = 3
for ix, i in stack_events.loc[stack_events.combo_id == id_id,:].drop_duplicates(["cid"]).iterrows():
    x = i["long"]
    y = i["lat"]
    start_angle = i["start_angle"]
    end_angle = i["end_angle"]
    radius = i["plot_radius"]
    pac_2 = mpatches.Wedge(center = [x, y], r = radius, theta2=-270 - start_angle, theta1=-270-end_angle, alpha = 0.3)
    pac_2.set_color('cyan')
    plt.gca().add_patch(pac_2)


colors = [palette_pastel[x] for x, i in enumerate(stack_events.tstamp.dt.date.unique())]
for x, i in enumerate([stack_events.loc[((stack_events.combo_id == id_id) &\
                    (stack_events.tstamp.dt.date == date)),["sector_centroid_lon", "sector_centroid_lat"]] for date in stack_events.tstamp.dt.date.unique()]):
    plt.plot(i["sector_centroid_lon"], i["sector_centroid_lat"],
            c = "black", alpha = 0.6, linewidth = 1)
plt.axis('equal')
plt.savefig("algo_idea.png")



pewpew = []
for index, combo in enumerate(list(list(i) for i in combinations(test['msisdn'].unique(), 2))[0:10000]):
    # print(test.loc[test['msisdn']==combo[0],:].shape, test.loc[test['msisdn'] == combo[1],:].shape)
    event_frame = test.loc[test['msisdn'].isin(combo),:].copy()
    event_frame.loc[:,"combo_id"] = index
    # print(event_frame.head())
    # event_frame.to_csv(os.path.join("combos", str(index) + ".csv"))

    event_frame = event_frame.drop(['lac', 'imei', 'max_dist', 'event_type',
                'lat_prev', 'long_prev', 'previous_timestamp', 'previous_status_distance',
                'segment_lat', 'segment_lon',
                'station_angle', 'sector_centroid_shift'], axis = 1).\
                sort_values(['combo_id', 'tstamp'], ascending = False)

    event_frame["msisdn_lag"] = event_frame.groupby(["combo_id"])["msisdn"].shift(-1)
    event_frame.loc[event_frame["msisdn_lag"] != event_frame["msisdn"],:]

    event_frame["tstamp_lag"] = event_frame.groupby(["combo_id"])["tstamp"].shift(-1)
    event_frame["sector_centroid_lat_lag"] = event_frame.groupby(["combo_id"])["sector_centroid_lat"].shift(-1)
    event_frame["sector_centroid_lon_lag"] = event_frame.groupby(["combo_id"])["sector_centroid_lon"].shift(-1)

    event_frame["vendor_all"] = event_frame["vendor"].str.cat(sep = ", ")
    event_frame["platform_all"] = event_frame["platform"].str.cat(sep = ", ")
    event_frame["type_all"] = event_frame["type"].str.cat(sep = ", ")
    event_frame["event_type_all"] = event_frame["event_symbol"].str.cat(sep = ", ")

    # event_frame["lat_change"] = event_frame["sector_centroid_lat"] - event_frame["sector_centroid_lat_lag"]
    # event_frame["lat_change"] = event_frame["sector_centroid_lat"] - event_frame["sector_centroid_lat_lag"]

    event_frame.loc[~event_frame["sector_centroid_lat_lag"].isnull(),"path_traveled"] = event_frame.loc[~event_frame["sector_centroid_lat_lag"].isnull(),:].\
                                            apply(lambda x: geodesic((x["sector_centroid_lat"],
                                                                        x["sector_centroid_lon"]),
                                                                    (x["sector_centroid_lat_lag"],
                                                                    x["sector_centroid_lon_lag"])).kilometers, axis = 1)
    event_frame = event_frame.dropna()
    event_frame["path_hours"] = (event_frame['tstamp'] - event_frame['tstamp_lag']).dt.seconds/3600



    event_frame["path_speed"] = (event_frame["path_traveled"] / event_frame["path_hours"])
    # stack_events[stack_events["path_speed"] != np.inf]["path_speed"].mean()

    latmin = event_frame.groupby(['date'])['sector_centroid_lat'].min()
    latmin.name = "lat_min"
    latmax = event_frame.groupby(['date'])['sector_centroid_lat'].max()
    latmax.name = "lat_max"
    lonmin = event_frame.groupby(['date'])['sector_centroid_lon'].min()
    lonmin.name = "lon_min"
    lonmax = event_frame.groupby(['date'])['sector_centroid_lon'].max()
    lonmax.name = "lon_max"

    location_data["binding_box_diag"] = location_data.apply(lambda x: geodesic((x["lat_min"],
                                x["lon_min"]),
                            (x["lat_max"],
                            x["lon_max"])).meters, axis = 1)

    bbox_25p = location_data["binding_box_diag"].describe()["25%"]
    bbox_50p = location_data["binding_box_diag"].describe()["50%"]
    bbox_75p = location_data["binding_box_diag"].describe()["75%"]
    bbox_mean = location_data["binding_box_diag"].describe()["mean"]

    speed_25p = event_frame[event_frame["path_speed"] != np.inf]["path_speed"].describe()["25%"]
    speed_50p = event_frame[event_frame["path_speed"] != np.inf]["path_speed"].describe()["50%"]
    speed_75p = event_frame[event_frame["path_speed"] != np.inf]["path_speed"].describe()["75%"]
    speed_mean = event_frame[event_frame["path_speed"] != np.inf]["path_speed"].describe()["mean"]

    nums = list(set(event_frame["msisdn"].unique().tolist() + event_frame["msisdn_lag"].unique().tolist()))
    nums.sort()
    if len(nums) > 1:
        pair = pd.Series({
                    "msisdn" : nums[0],
                    "msisdn_lag" : nums[1],
                    "bbox_25p" : bbox_25p,
                    "bbox_50p" : bbox_50p,
                    "bbox_75p" : bbox_75p,
                    "bbox_mean" : bbox_mean,

                    "speed_25p" : speed_25p,
                    "speed_50p" : speed_50p,
                    "speed_75p" : speed_75p,
                    "speed_mean" : speed_mean
                    })
        pewpew.append(pair)
    # stack_events = pd.concat([stack_events, pair], axis = 0)
    # stack_events = pd.concat([stack_events, event_frame], axis = 0)


pd.DataFrame(pewpew).head()







(datetime.now() - datetime.strptime('26092018', '%d%m%Y')).days
len(list(combinations(test['msisdn'].unique(), 2)))
