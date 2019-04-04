import requests
from fake_useragent import UserAgent
ua = UserAgent()


# 
# PARAMS = {"r": entry["reporter_id"],
#                   "px": entry["classif"],
#                   "ps": entry["year"],
#                   "p": "all",
#                   "rg": entry["regime"],
#                   "cc": entry["agg_level"],
#                   "max": 250000,
#                   "fmt": "json",
#                   "type": "C",
#                   "freq": "A",
#                   "head": "M"
#                   }

headers = requests.utils.default_headers()
headers['User-Agent'] = ua.random

request_bod = requests.get(url=API_ENDPOINT, params=PARAMS, headers=headers)

# https://xn--80az8a.xn--d1aqf.xn--p1ai/%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D1%82%D0%B8%D0%BA%D0%B0/grapi/v1/geoObjects
