from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():

    # making sparksession object
    conf = SparkConf().setAppName('Covid-19')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    spark = SparkSession(sc)

    # load data into spark dataframe
    ####################################
    df = spark.read.format("csv").option("header", "true").load(
        "time_series_19-covid-Confirmed_archived_0325.csv")

    # prepare data
    ####################################
    df = df.filter(F.col("Country/Region") == "Australia")
    columns_to_drop = ['Country/Region', 'Province/State', 'Lat', 'Long', '1/22/20', '1/23/20', '1/24/20',
                       '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20', '1/30/20', '1/31/20', '3/22/20', '3/23/20']
    df = df.drop(*columns_to_drop)
    # sum up all rows data into 1 row
    # df.select(F.sum("2/1/20").alias('2/1/20'),
    # F.sum("2/2/20")).show()  # its todo
    new_df = df.select([F.sum(value).alias(str(index))
                        for index, value in enumerate(df.columns)]).show()
    new_df.show()
    # if want to save to a csv
    # new_df.write.option('header', 'true').csv('data.csv')

    # print(df.agg({"3/21/20": "sum"}).collect())
    # df.groupBy().sum().show()
    # df.show()

    # clean data
    ####################################

    # linear regression
    ####################################

    # visualize results
    ####################################


if __name__ == "__main__":
    main()
