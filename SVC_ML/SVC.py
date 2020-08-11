import mlflow
import json
from pyspark.ml.classification import LinearSVC
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.linalg import Vector
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('Project').getOrCreate()

dataset=spark.read.csv("items.tbl", inferSchema = True, header = True, sep = '|')
dataset.createTempView("items")

df = spark.read.json("clicks1.json")
df.createTempView("clicks")


q="SELECT logs.wl_customer_id as wl_customer_id, i.i_category_id FROM(select wl_customer_id, wl_item_id from clicks lateral view json_tuple('wl_customer_id','wl_item_id') l where wl_customer_id is not null) logs, items i WHERE logs.wl_item_id = i.i_item_id AND i.i_category_id IS NOT NULL"

df_2 = spark.sql(q)
df_2.createTempView("web_clickstreams")
q2="select wl_customer_id, sum(case when i_category_id=0 then 1 else 0 end) as clicks_in_0,sum(case when i_category_id=1 then 1 else 0 end) as clicks_in_1,sum(case when i_category_id=2 then 1 else 0 end) as clicks_in_2,sum(case when i_category_id=3 then 1 else 0 end) as clicks_in_3,sum(case when i_category_id=4 then 1 else 0 end) as clicks_in_4,sum(case when i_category_id=5 then 1 else 0 end) as clicks_in_5,sum(case when i_category_id=6 then 1 else 0 end) as clicks_in_6,sum(case when i_category_id=7 then 1 else 0 end) as clicks_in_7,sum(case when i_category_id=8 then 1 else 0 end) as clicks_in_8,sum(case when i_category_id=9 then 1 else 0 end) as clicks_in_9,sum(case when i_category_id=10 then 1 else 0 end) as clicks_in_10,sum(case when i_category_id=11 then 1 else 0 end) as clicks_in_11,sum(case when i_category_id=12 then 1 else 0 end) as clicks_in_12,sum(case when i_category_id=13 then 1 else 0 end) as clicks_in_13,sum(case when i_category_id=14 then 1 else 0 end) as clicks_in_14,sum(case when i_category_id=15 then 1 else 0 end) as clicks_in_15,sum(case when i_category_id=16 then 1 else 0 end) as clicks_in_16,sum(case when i_category_id=17 then 1 else 0 end) as clicks_in_17,sum(case when i_category_id=18 then 1 else 0 end) as clicks_in_18,sum(case when i_category_id=19 then 1 else 0 end) as clicks_in_19 from web_clickstreams group by wl_customer_id"
df_3 = spark.sql(q2)
df_3.createTempView("category_clicks")
q3="SELECT CASE WHEN clicks_in_0 > 3.871510156577754 THEN 1.0 ELSE 0.0 END AS label,clicks_in_1,clicks_in_2,clicks_in_3,clicks_in_4,clicks_in_5,clicks_in_6,clicks_in_7,clicks_in_8,clicks_in_9,clicks_in_10,clicks_in_11,clicks_in_12,clicks_in_13,clicks_in_14,clicks_in_15,clicks_in_16,clicks_in_17,clicks_in_18,clicks_in_19 FROM category_clicks"
df_4 =spark.sql(q3)

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["clicks_in_1", "clicks_in_2", "clicks_in_3", "clicks_in_4", "clicks_in_5", "clicks_in_6" , "clicks_in_7", "clicks_in_8", "clicks_in_9" , "clicks_in_10", "clicks_in_11", "clicks_in_12" , "clicks_in_13", "clicks_in_14", "clicks_in_15" , "clicks_in_16", "clicks_in_17", "clicks_in_18" , "clicks_in_19"],outputCol ="features") 
vd = assembler.transform(df_4)

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(vd)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(vd)

training, test = vd.randomSplit([0.6, 0.4])

lsvc = LinearSVC(maxIter=10, regParam = 0.001, tol = 0.01, aggregationDepth = 2)
model = lsvc.fit(training)

print ("Coefficients = "+ str(model.coefficients))

print ("Intercept = "+ str(model.intercept))


params = model.explainParams()
print(params)

mlflow.log_param("maxIter",10)
mlflow.log_param("regParam", 0.001)
mlflow.log_param("tol", 0.01)
mlflow.log_param("aggregationDepth", 2)

#mlflow.log_metric("Coefficients", str(model.coefficients))
mlflow.log_metric("Intercept", round(model.intercept, 8))



