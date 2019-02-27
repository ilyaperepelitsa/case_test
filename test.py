import pandas as pd
from datetime import datetime


pd.set_option('display.float_format', lambda x: '%.3f' % x)


facts = pd.read_excel("01_Facts.xlsx", header = None)
event_type_data = pd.read_excel("04_event_type.xlsx")
event_type_data = event_type.rename(columns = {"Номер":"event_type", "Обозначение":"event_symbol", "Описание":"event_description"})
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

# devices
# for i in devices:
#     print(i.shape)


test.merge(event_type_data, on = "event_type")

event_type_data.event_type.values
test.event_type.values


.astype({"event_type" : "object"})

pd.Series(test.event_type.values).isin(event_type_data.event_type.values).any()


test.merge(event_type)

test.
event_type.info()
test.info()

test.imei.isnull().sum()
test.msisdn.isnull().sum()
test







test.describe()
test.info()

test["msisdn"].unique().shape[0]
test.sort_values(["msisdn", "tstamp"])
