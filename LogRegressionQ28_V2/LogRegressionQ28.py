import mlflow
import mlflow.spark
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, IndexToString, VectorIndexer
from pyspark.ml.linalg import Vector
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('Project').getOrCreate()

dataset=spark.read.csv("reviews.tbl", inferSchema = True, header = True, sep = '|')

dataset.createTempView("product_reviews")
q="SELECT CASE pr_rating WHEN 1 THEN '0' WHEN 2 THEN '0' WHEN 3 THEN '1' WHEN 4 THEN '3' WHEN 5 THEN '3' END AS pr_r_rating, pr_content FROM product_reviews WHERE pmod(pr_review_id, 5) IN (1,2,3)"
df = spark.sql(q).toDF("label", "sentence")
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(df)
hashingTF = HashingTF(inputCol="words",outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol = "rawFeatures",outputCol = "userFeatures")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

df = rescaledData.select(rescaledData["label"].cast("double"),(rescaledData["userFeatures"]))

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["userFeatures"],outputCol="features")
output = assembler.transform(df)

training, test = output.randomSplit([0.6, 0.4])
lr = LogisticRegression(maxIter=10, regParam=0.1, elasticNetParam=0.00001)

model = lr.fit(training)
trainingSummary = model.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# for multiclass, we can inspect metrics on a per-label basis
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))

print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))

print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

mlr = LogisticRegression(maxIter=10, regParam=0.1, elasticNetParam=0.00001, family="multinomial")

mlrModel = mlr.fit(training)

# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: ", mlrModel.coefficientMatrix)
print("Multinomial intercepts: " , mlrModel.interceptVector)
   
mlflow.log_param("maxIter", 10)
mlflow.log_param("regParam", 0.1)
mlflow.log_param("elasticNetParam", 0.00001)
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("FPR", falsePositiveRate)
mlflow.log_metric("TPR", truePositiveRate)
mlflow.log_metric("F-measure", fMeasure)
mlflow.log_metric("Precision", precision)
mlflow.log_metric("Recall", recall)

params = model.explainParams()
print(params)

#plt.figure (figsize=(5,5))
#plt.plot([0,1],[0,1], 'r--')
#plt.plot(roc['FPR'], roc['TPR'])
#plt.xlabel('FPR')
#plt.ylabel('TPR')
#plt.show()
#plt.plot(pr['recall'], pr['precision'])
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.show()

