import pandas as pd


facts = pd.read_excel("01_Facts.xlsx", header = None)
event_type = pd.read_excel("04_event_type.xlsx")
event_type.to_dict("records")
