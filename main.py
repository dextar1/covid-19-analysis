from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# making sparksession object
spark = SparkSession.builder.appName('Covid-19').getOrCreate()

# load data into spark dataframe
####################################
df = spark.read.format("csv").option("header", "true").load("time_series_19-covid-Confirmed_archived_0325.csv")


# prepare data
####################################
df = df.filter(col("Country/Region") == "Australia")
columns_to_drop = ['Country/Region','Province/State', 'Lat', 'Long', '1/22/20','1/23/20','1/24/20','1/25/20','1/26/20','1/27/20','1/28/20','1/29/20','1/30/20','1/31/20','3/22/20','3/23/20']
df = df.drop(*columns_to_drop)
# df.show()


# clean data
####################################



# linear regression
####################################


# visualize results
####################################