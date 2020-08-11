import mlflow
from pyspark.ml.classification import NaiveBayes
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vector
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName('Project').getOrCreate()

dataset=spark.read.csv("reviews.tbl", inferSchema = True, header = True, sep = '|')

dataset.createTempView("product_reviews")
q="SELECT CASE pr_rating WHEN 1 THEN '0' WHEN 2 THEN '0' WHEN 3 THEN '1' WHEN 4 THEN '2' WHEN 5 THEN '2' END AS pr_r_rating, pr_content FROM product_reviews WHERE pmod(pr_review_id, 5) IN (1,2,3)"
df = spark.sql(q).toDF("label","sentence")
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(df)
hashingTF = HashingTF(inputCol="words",outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol = "rawFeatures",outputCol = "features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

df = rescaledData.select(rescaledData["label"].cast("double"),(rescaledData["features"]))
training, test = df.randomSplit([0.6, 0.4])

nb = NaiveBayes(smoothing=1.0, modelType='multinomial')
model = nb.fit(training)
predictions= model.transform(test)
result = model.transform(test).head()
rp=result.rawPrediction
pr=result.probability
print(rp)
print(pr)

pi = model.pi
theta =model.theta
print(pi)
print(theta)


evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="accuracy")

accuracy=evaluator.evaluate(predictions)
print ("Test set accuracy = "+ str(accuracy))
params = model.explainParams()
print(params)

mlflow.log_param("smoothing", 1.0)
mlflow.log_param("modelType", 'multinomial')
mlflow.log_metric("Test set accuracy", accuracy)
