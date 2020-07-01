from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType

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
    df = df.select([F.sum(value).alias(str(index))
                        for index, value in enumerate(df.columns)])
    # transpose the dataframe
    df_p = df.toPandas().transpose().reset_index()
    df_p.rename(columns={0:'Infections'})
    df_s = spark.createDataFrame(df_p)
    df_s = df_s.select(F.col('index'), F.col("0").alias("Infections"))

    # linear regression
    ####################################
    df_s = df_s.withColumn("index_double", df_s['index'].cast(DoubleType()))
    featureassembler = VectorAssembler(inputCols=["index_double"],outputCol="new_index")
    output=featureassembler.transform(df_s)
    finalized_data = output.select("new_index","Infections")
    train_data,test_data = finalized_data.randomSplit([0.75,0.25])
    regressor = LinearRegression(featuresCol='new_index', labelCol='Infections')
    regressor = regressor.fit(train_data)
    pred_results=regressor.evaluate(test_data)
    pred_results.predictions.show(40)

    # visualize results
    ####################################


if __name__ == "__main__":
    main()
