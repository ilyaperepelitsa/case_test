import findspark
from pyspark import SparkContext
# import SparkContext from pyspark.sql

from pyspark import SparkContext, SparkSession


import SparkSession
findspark.init()

sc = SparkContext(appName="MyFirstApp")
spark = SparkSession(sc)
print("Hello World!")
sc.close() #closing the spark session
