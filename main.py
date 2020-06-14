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

# clean data
####################################


# prepare data
####################################


# linear regression
####################################


# visualize results
####################################