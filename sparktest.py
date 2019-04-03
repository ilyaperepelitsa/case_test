import findspark from pyspark
from findspark import findspark
import SparkContext from pyspark.sql
import SparkSession
findspark.init()

sc = SparkContext(appName="MyFirstApp")
spark = SparkSession(sc)
print("Hello World!")
sc.close() #closing the spark session
