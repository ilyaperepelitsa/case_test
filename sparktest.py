import findspark
from pyspark import SparkContext
# import SparkContext from pyspark.sql

from pyspark import SparkContext
from pyspark.sql import SparkSession

# import SparkSession
findspark.init()

sc = SparkContext(appName="MyFirstApp")
spark = SparkSession(sc)
print("Hello World!")
sc.stop() #closing the spark session


data_hetero = sc.parallelize([
    ("Ferrari", "fast"),
    {"Porsche" : 100000},
    ["spain", 'visited', 4504]
])
