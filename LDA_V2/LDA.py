import mlflow
from pyspark import Row
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType
from pyspark.sql.functions import col

spark = SparkSession.builder.appName('Project').getOrCreate()

dataset=spark.read.csv("reviews.tbl", inferSchema = True, header = True, sep = '|')

dataset.createTempView("product_reviews")
q="SELECT CASE pr_rating WHEN 1 THEN 'NEG' WHEN 2 THEN 'NEG' WHEN 3 THEN 'NEU' WHEN 4 THEN 'POS' WHEN 5 THEN 'POS' END AS pr_r_rating, pr_content FROM product_reviews WHERE pmod(pr_review_id, 5) IN (1,2,3)"
df = spark.sql(q).toDF("label", "sentence")
tokenizer = Tokenizer(inputCol="sentence", outputCol="tokens")
wordsData = tokenizer.transform(df)
# remove stop words
remover = StopWordsRemover(inputCol="tokens", outputCol="words")
cleaned = remover.transform(wordsData)

# vectorize
cv = CountVectorizer(inputCol="words", outputCol="features")
count_vectorizer_model = cv.fit(cleaned)
result = count_vectorizer_model.transform(cleaned)

#corpus = result.select('vectors').rdd.map(lambda x: Row (x[0])).toDF()
#corpus=corpus.select(col("_1").alias("features"))

ldaModel = LDA(k=4, maxIter =100)
model = ldaModel.fit(result)
# extracting topics
topics = model.describeTopics(maxTermsPerTopic=10)
# extraction vocabulary
vocabulary = count_vectorizer_model.vocabulary

ll = model.logLikelihood(result)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))


lp = model.logPerplexity(result)
print("The upper bound on perplexity: " + str(lp))


topics = model.describeTopics(3)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)

#transformed = model.transform(corpus)
#transformed.show(truncate=False)

vocabulary = model.vocabSize()
params = model.explainParams()
print(params)
print(vocabulary)


mlflow.log_param("k", 10)
mlflow.log_param("maxIter", 10)
mlflow.log_metric("The lower bound on the log likelihood", ll)
mlflow.log_metric("The upper bound on perplexity", lp)

