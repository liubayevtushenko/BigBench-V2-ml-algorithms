import mlflow
from mlflow.tracking.client import MlflowClient
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt
import numpy as np
from pyspark.sql import Row
from pyspark.rdd import RDD
import sys
sys.setrecursionlimit(1500)

spark = SparkSession.builder.appName('Project').getOrCreate()

store_sales=spark.read.csv("store_sales.tbl", inferSchema = True, header = True, sep = '|')
items=spark.read.csv("items.tbl", inferSchema = True, header = True, sep = '|')

store_sales.createTempView('store_sales')
items.createTempView('items')

q="SELECT ss.ss_customer_id AS cid, count(CASE WHEN i.i_class_id=1  THEN 1 ELSE NULL END) AS id1,count(CASE WHEN i.i_class_id=3  THEN 1 ELSE NULL END) AS id3,count(CASE WHEN i.i_class_id=5  THEN 1 ELSE NULL END) AS id5,count(CASE WHEN i.i_class_id=7  THEN 1 ELSE NULL END) AS id7, count(CASE WHEN i.i_class_id=9  THEN 1 ELSE NULL END) AS id9,count(CASE WHEN i.i_class_id=11 THEN 1 ELSE NULL END) AS id11,count(CASE WHEN i.i_class_id=13 THEN 1 ELSE NULL END) AS id13,count(CASE WHEN i.i_class_id=15 THEN 1 ELSE NULL END) AS id15,count(CASE WHEN i.i_class_id=2  THEN 1 ELSE NULL END) AS id2,count(CASE WHEN i.i_class_id=4  THEN 1 ELSE NULL END) AS id4,count(CASE WHEN i.i_class_id=6  THEN 1 ELSE NULL END) AS id6,count(CASE WHEN i.i_class_id=12 THEN 1 ELSE NULL END) AS id12, count(CASE WHEN i.i_class_id=8  THEN 1 ELSE NULL END) AS id8,count(CASE WHEN i.i_class_id=10 THEN 1 ELSE NULL END) AS id10,count(CASE WHEN i.i_class_id=14 THEN 1 ELSE NULL END) AS id14,count(CASE WHEN i.i_class_id=16 THEN 1 ELSE NULL END) AS id16 FROM store_sales ss INNER JOIN items i ON ss.ss_item_id = i.i_item_id WHERE i.i_category_name IN ('cat#01','cat#02','cat#03','cat#04','cat#05','cat#06','cat#07','cat#08','cat#09','cat#10','cat#11','cat#12','cat#013','cat#14','cat#15') AND ss.ss_customer_id IS NOT NULL GROUP BY ss.ss_customer_id HAVING count(ss.ss_item_id) > 3"

df = spark.sql(q)


assembler = VectorAssembler(inputCols=["cid", "id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8", "id9", "id10"],outputCol ="features")

vd = assembler.transform(df)
cost = list()

#with mlflow.start_run():
  #for k in range (2,15):
kmeans =       KMeans().setK(4).setInitSteps(5).setMaxIter(20).setTol(0.01).setFeaturesCol('features').setPredictionCol('prediction')

model = kmeans.fit(vd)
predictions = model.transform(vd)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)



mlflow.log_param("maxIter", 20)
mlflow.log_param("initSteps", 5)
mlflow.log_param("tol",0.01)
mlflow.log_param("k", 4)

mlflow.log_metric("Silhouette_with_squared_euclidean_distance", silhouette)

centers = model.clusterCenters()
params = model.explainParams()

print(params)

print("Cluster Centers: ")
for center in centers:
    print(center)

# Evaluate clustering by computing Silhouette score

print("Silhouette with squared euclidean distance = " + str(silhouette))

mlflow.log_param("maxIter", 20)
mlflow.log_param("initSteps", 5)
mlflow.log_param("tol",0.01)
mlflow.log_param("k", 12)

mlflow.log_metric("Silhouette_with_squared_euclidean_distance", silhouette)
