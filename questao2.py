import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, count, to_date, to_timestamp, year, month, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, ArrayType, FloatType
import nltk
from nltk import ngrams
from textblob import TextBlob

nltk.download('punkt')

spark = SparkSession.builder \
    .appName("EiffelTowerReviews") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

schema = StructType([
    StructField("_id", StructType([StructField("$oid", StringType(), True)])),
    StructField("title", StringType(), True),
    StructField("bubbleCount", IntegerType(), True),
    StructField("createdAt", StringType(), True),
    StructField("text", StringType(), True),
    StructField("query", StringType(), True),
    StructField("author", StructType([
        StructField("memberSince", StringType(), True),
        StructField("reviews", StringType(), True),
        StructField("ratings", StringType(), True),
        StructField("postForum", IntegerType(), True),
        StructField("helpfulVotes", IntegerType(), True),
        StructField("level", StringType(), True)
    ])),
    StructField("collectedAt", StructType([StructField("$date", StringType(), True)]))
])

file_path = "datasets/eiffel-tower-reviews.json"
df = spark.read.schema(schema).json(file_path)

df = df.withColumn("date", to_date(to_timestamp(col("createdAt"), "MMM dd, yyyy")))

words_df = df.select(explode(split(col("text"), "\\s+")).alias("word"))
word_count_df = words_df.groupBy("word").count().orderBy("count", ascending=False)
word_count_df.show(10)

def extract_phrases(text):
    if text is None:
        return []
    words = nltk.word_tokenize(text)
    bigrams = ngrams(words, 2)
    trigrams = ngrams(words, 3)
    return [' '.join(gram) for gram in bigrams] + [' '.join(gram) for gram in trigrams]

extract_phrases_udf = udf(extract_phrases, ArrayType(StringType()))
phrases_df = df.withColumn("phrases", explode(extract_phrases_udf(col("text"))))
phrase_count_df = phrases_df.groupBy("phrases").count().orderBy("count", ascending=False)
phrase_count_df.show(10)

from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA

words_df = df.filter(col("text").isNotNull()).withColumn("words", split(col("text"), "\\s+"))
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=10000, minDF=5)
cv_model = cv.fit(words_df)
count_vectors = cv_model.transform(words_df)

lda = LDA(k=5, maxIter=10)
lda_model = lda.fit(count_vectors)

topics = lda_model.describeTopics(5)
vocab = cv_model.vocabulary
topics_rdd = topics.rdd.map(lambda row: row['termIndices']).map(lambda idx_list: [vocab[idx] for idx in idx_list])
for idx, topic in enumerate(topics_rdd.collect()):
    print(f"Topic {idx + 1}: {', '.join(topic)}")


df_by_month = df.withColumn("year", year("date")).withColumn("month", month("date"))
reviews_by_month = df_by_month.groupBy("year", "month").count().orderBy("year", "month")
reviews_by_month.show()

def get_sentiment(text):
    if text is None:
        return 0.0
    return TextBlob(text).sentiment.polarity

sentiment_udf = udf(get_sentiment, FloatType())
df = df.withColumn("sentiment", sentiment_udf(col("text")))

df.groupBy("sentiment").count().orderBy("sentiment").show()
