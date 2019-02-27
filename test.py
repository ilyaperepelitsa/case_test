import pandas as pd
from datetime import datetime


pd.set_option('display.float_format', lambda x: '%.3f' % x)


facts = pd.read_excel("01_Facts.xlsx", header = None)
event_type = pd.read_excel("04_event_type.xlsx")
event_type
# event_type.to_dict("records")
test = pd.read_csv("02_Data_test.csv", sep = ";", dtype = {"imei" : "object",
                                                            "lac" : "object",
                                                            "cid" : "object",
                                                            "msisdn" : "object",
                                                            "event_type" : "object"})
test['tstamp'] = pd.to_datetime(test['tstamp'], unit='ms')
# test

# test.groupby(["cid"])["max_dist"].describe()
devices = pd.read_csv("03_устройства.csv", sep = "\"*,\"*")
devices = devices.applymap(lambda x: x.replace("\"", ""))

devices.columns = devices.columns.str.replace("\"", "")

test["tac"] = test["imei"].str.extract("^(\d{8})")
test = test.merge(devices, on = "tac", how = "left")

# devices
# for i in devices:
#     print(i.shape)

test["tac"] = test["imei"].str.extract("^(\d{8})")
test = test.merge(devices, on = "tac", how = "left")
test.imei.isnull().sum()
test.msisdn.isnull().sum()
test







test.describe()
test.info()

test["msisdn"].unique().shape[0]
test.sort_values(["msisdn", "tstamp"])
