from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from matplotlib import pyplot as plt

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
                       '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20', '1/30/20', '1/31/20'] # '3/22/20', '3/23/20' will be the test data, will remove bit later
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
    # df_s = df_s.withColumn("infections_double", df_s['Infections'].cast(DoubleType()))
    featureassembler = VectorAssembler(inputCols=["index_double"],outputCol="new_index")
    output=featureassembler.transform(df_s)
    full_data = output.select("new_index","Infections")
    test_data = full_data.where(F.col('index_double') > 49)
    train_data = full_data.where(F.col('index_double') < 50)
    train_data.show(50)
    test_data.show()
    regressor = LinearRegression(featuresCol='new_index', labelCol='Infections')
    regressor = regressor.fit(train_data)
    pred_results=regressor.evaluate(test_data)
    pred_results.predictions.show(60)
    print("Coefficients: " + str(regressor.coefficients))
    print("Intercept: " + str(regressor.intercept))

    trainingSummary = regressor.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    # visualize results
    ####################################
    actual_values = pred_results.predictions.select('new_index','Infections').collect()
    predicted_values = pred_results.predictions.select('new_index','prediction').collect()
    def dfToList(df):
        av_list = []
        for row in df:
            l = []
            for val in row:
                if type(val) is DenseVector:
                    l.append(val.values[0])
                else:
                    l.append(val)
            av_list.append(l)
        return av_list
    av_list = dfToList(actual_values)
    pv_list = dfToList(predicted_values)
    x1 = [c[0] for c in av_list]
    y1 = [c[1] for c in av_list]
    x2 = [c[0] for c in predicted_values]
    y2 = [c[1] for c in predicted_values]
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.show()


if __name__ == "__main__":
    main()
