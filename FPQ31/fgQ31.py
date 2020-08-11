from pyspark.sql import SQLContext
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql import Row
import mlflow
import json

spark = SparkSession.builder.appName('Project').getOrCreate()


dataset=spark.read.csv("items.tbl", inferSchema = True, header = True, sep = '|')
dataset.createTempView("items")


#df = spark.read.json("clicks1.json")
#df.createTempView("clicks")


#q="SELECT collect_set(logs.wl_customer_id) as items, i.i_category_id FROM(select wl_customer_id, wl_item_id from clicks lateral view json_tuple('wl_customer_id','wl_item_id') l where wl_customer_id is not null) logs, items i WHERE logs.wl_item_id = i.i_item_id AND i.i_category_id IS NOT NULL group BY i_category_id"

#df = spark.sql(q)

store_sales=spark.read.csv("store_sales.tbl", inferSchema = True, header = True, sep = '|')

store_sales.createTempView('store_sales')


q="Select collect_set(ss_item_id) as items FROM store_sales s INNER JOIN items i ON (s.ss_item_id = i.i_item_id) GROUP BY s.ss_transaction_id"

df = spark.sql(q)


fpGrowth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.9, numPartitions = 2)
model = fpGrowth.fit(df)

param = model.explainParams()

print(param)

# Display frequent itemsets.
model.freqItemsets.show()

model.associationRules.show()

model.transform(df).show()

associationsRules = model.associationRules

associationsRules.createTempView("associationsRules")

mlflow.log_param("minSupport", 0.01)
mlflow.log_param("minConfidence", 0.9)
mlflow.log_param("numPartitions", 2)

