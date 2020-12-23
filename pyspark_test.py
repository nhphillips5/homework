import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, asc, desc, avg

spark = SparkSession.builder.appName("FinalPyspark").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('WARN')


files = ["randomint1.txt",
        "randomint2.txt",
        "randomint3.txt",
        "randomint4.txt",
        "randomint5.txt",
        "randomint6.txt",
        "randomint7.txt"]

integers = []
for f in files:
    integers.append(pd.read_csv(f, header = None))

ints = pd.DataFrame()

ints = ints.append(integers)

intsRdd = sqlContext.createDataFrame(ints)

#Show the largest integer
intsRdd.sort(col('0').desc()).show()

#Find the average of all integers.
mean = intsRdd.agg(avg(col('0')))
mean.show()

#report the number of unique integers
intsRdd.select('0').distinct().count()



