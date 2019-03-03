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

pd.set_option('display.float_format', lambda x: '%.7f' % x)


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
    # print(type(lat2))
    return lat2, lon2


def sector_angle_mid(data):
    angle_mid = (data["end_angle"] + data["start_angle"]) / 2
    if angle_mid > 360:
        angle_mid = angle_mid - 180
    # print(type(lat2))
    return angle_mid

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

# test["segment_lat"], test["segment_lon"] = test.apply(circle_segment, axis = 1)

test["segment_lat"], test["segment_lon"] = test.apply(circle_segment, axis = 1).apply(lambda x: x[0]), test.apply(circle_segment, axis = 1).apply(lambda x: x[1])

test["plot_radius"] = test.apply(lambda x: math.hypot(x["segment_lon"] - x["long"], x["segment_lat"] - x["lat"]), axis = 1)
test["plot_radius_2"] = test.apply(lambda x: math.sqrt((x["segment_lon"] - x["long"])**2 + (x["segment_lat"] - x["lat"])**2), axis = 1)

test["station_angle"] = test.end_angle - test.start_angle
# test["station_angle"][test["station_angle"] < 0] += 360
test["station_angle"][test["station_angle"] < 0] = test["station_angle"][test["station_angle"] < 0] + 360

test["sector_centroid_shift"] = test.apply(lambda x: (2*x["max_dist"] * np.sin(math.radians(x["station_angle"]))) / (3*math.radians(x["station_angle"])),axis = 1)
test["sector_centroid_lat"], test["sector_centroid_lon"] = test.apply(sector_centroid, axis = 1).apply(lambda x: x[0]), test.apply(sector_centroid, axis = 1).apply(lambda x: x[1])


test["sector_centroid_shift"]
np.sin(math.degrees(45))

# pd.concat([test["plot_radius"], test["plot_radius_2"]], axis = 1)
pd.concat([test["sector_centroid_shift"], test["max_dist"]], axis = 1)



# pd.concat([test["start_angle"], test.apply(sector_angle_mid, axis = 1), test["end_angle"]], axis = 1)
# plt.hist(np.log((2*test["max_dist"] * np.sin(test["station_angle"])) / 3*np.sin(test["station_angle"])))
# pd.concat([test["start_angle"], test["station_angle"], test["end_angle"]], axis = 1)


test[["sector_centroid_lat","sector_centroid_lon"]]


test.head().apply(lambda x: x["max_dist"]/1000, axis = 1)


geodesic(kilometers=data["max_dist"]/1000)
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
    # angle = i["end_angle"] - i["start_angle"]
    start_angle = i["start_angle"]
    end_angle = i["end_angle"]
    radius = i["plot_radius"]


    # pac = mpatches.Wedge(center = [x, y], r = radius, theta1=start_angle + 90, theta2=end_angle)
    # pac = mpatches.Wedge(center = [x, y], r = radius, theta1=0, theta2=15)
    # pac_2 = mpatches.Wedge(center = [x + 0.1, y + 0.1], r = radius, theta1=0, theta2=35)
    pac_2 = mpatches.Wedge(center = [x, y], r = radius, theta2=-270 - start_angle, theta1=-270-end_angle, alpha = 0.3)
    # pac_23 = [mpatches.Wedge(center = [x, y], r = radius*1.2, theta1=0, theta2=0, alpha = 0.7),
    #             mpatches.Wedge(center = [x, y], r = radius*1.5, theta1=0, theta2=0, alpha = 0.7),
    #             mpatches.Wedge(center = [x, y], r = radius*2, theta2=-270, theta1=-270-90,alpha = 0.7)]
    # plt.gca().add_patch(pac)
    plt.gca().add_patch(pac_2)
# for plo in pac_23:
#     plt.gca().add_patch(plo)
#     plo.set_color("orange")
    pac_2.set_color(new_cmap(ix))
    # plt.scatter(i["sector_centroid_lon"], i["sector_centroid_lat"])
plt.axis('equal')
plt.show()



plt.savefig("pewpew.png")

i["sector_centroid_lon"], i["sector_centroid_lat"]
x, y
start_angle, end_angle

plt.figure(figsize=(20,10))
# for x, i in test.drop_duplicates("cid").head().iterrows():
#     # print(i["cid"])
x = 1
y = 1
# angle = i["end_angle"] - i["start_angle"]
start_angle = math.degrees(0)
end_angle = math.degrees(90)
radius = 1

pac_2 = mpatches.Wedge(center = [x, y], r = radius, theta1=math.radians(start_angle), theta2=math.radians(end_angle), alpha = 0.6)
plt.gca().add_patch(pac_2)
pac_2.set_color('cyan')
# plt.scatter(x = test.drop_duplicates("cid").head()["long"], y = test.drop_duplicates("cid").head()["lat"])
# plt.scatter(x = test.drop_duplicates("cid").head()["sector_centroid_lon"], y = test.drop_duplicates("cid").head()["sector_centroid_lat"])
plt.axis('equal')
plt.show()






plt.figure(figsize=(20,10))
# for x, i in test.drop_duplicates("cid").iterrows():
    # print(i["cid"])

i = test.loc[test.index[0],:]
x = i["long"]
y = i["lat"]
# angle = i["end_angle"] - i["start_angle"]
start_angle = i["start_angle"]
end_angle = i["end_angle"]
radius = i["plot_radius"]


# pac = mpatches.Wedge(center = [x, y], r = radius, theta1=start_angle + 90, theta2=end_angle)
# pac = mpatches.Wedge(center = [x, y], r = radius, theta1=0, theta2=15)
# pac_2 = mpatches.Wedge(center = [x + 0.1, y + 0.1], r = radius, theta1=0, theta2=35)
pac_2 = mpatches.Wedge(center = [x, y], r = radius, theta1=start_angle-120, theta2=end_angle-120, alpha = 0.6)
# plt.gca().add_patch(pac)
plt.gca().add_patch(pac_2)
pac_2.set_color('cyan')
# plt.scatter(x = test.drop_duplicates("cid").head()["long"], y = test.drop_duplicates("cid").head()["lat"])
plt.scatter(x = i["sector_centroid_lon"], y = i["sector_centroid_lat"])
plt.axis('equal')
plt.show()




pd.concat([test["sector_centroid_shift"], test["plot_radius"]], axis = 1)



test.drop_duplicates("cid")

plt.scatter(test.drop_duplicates("cid").loc[:,["sector_centroid_lat"]],
                test.drop_duplicates("cid").loc[:,["sector_centroid_lon"]])

plt.scatter(test.drop_duplicates("cid").loc[:,["lat"]],
                test.drop_duplicates("cid").loc[:,["long"]])

test.head().loc[:,["long", "sector_centroid_lon", "lat", "sector_centroid_lat", "plot_radius"]]







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
# datetime.strftime("26092018", "%d%m%y")
(datetime.now() - datetime.strptime('26092018', '%d%m%Y')).days
