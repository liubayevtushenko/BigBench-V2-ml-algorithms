import mlflow
import json
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
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
#labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(vd)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
#featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",maxCategories=4).fit(vd)

# Train a DecisionTree model.
#dtr = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Split the data into training and test sets (30% held out for testing)
(training, test) = vd.randomSplit([0.7, 0.3])
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(training)
td = si_model.transform(training)

dt = DecisionTreeClassifier(maxDepth=7, maxBins=32, labelCol="indexed")
model = dt.fit(td)
print(model.numNodes)
print(model.depth)
print(model.featureImportances)
print(model.numFeatures)
print(model.numClasses)
print(model.toDebugString)

result = model.transform(test).head()

# Train a DecisionTree model.
#dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", predictionCol="prediction", probabilityCol="probability", rawPredictionCol="rawPrediction", maxBins = 4, maxDepth = 7, impurity = "gini")

# Chain indexers and tree in a Pipeline
#pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
#model = pipeline.fit(trainingData)

#params = model.explainParams()
#print(params)

# Make predictions.
predictions = model.transform(test)


# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

evaluator = RegressionEvaluator(
	        labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

#treeModel = model.stages[2]
# summary only
#print(treeModel)

mlflow.log_param("maxCategories", 4)

mlflow.log_metric("Accuracy", (accuracy))
mlflow.log_metric("Test Error", (1.0 - accuracy))

