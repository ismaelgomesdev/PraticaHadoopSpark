import socket
socket.setdefaulttimeout(1200)

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, hour, when, to_date, udf, to_timestamp, lower
from pyspark.sql.types import ArrayType, StringType
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

column_names = [
    "id", "text", "lat", "long", "flag", "id_str1", "id_str2", "created_at", "date", "user_id", "lang", 
    "location", "location_id", "metric1", "metric2", "metric3", "metric4", "metric5", "metric6", "metric7", 
    "metric8", "c21", "c22", "c23", "c24", "user_name", "user_id_str1", "user_id_str2", "c28", "metric9", 
    "user_created_at", "user_screen_name"
]


spark = SparkSession.builder \
    .appName("AnaliseTweets") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.cores", "4") \
    .config("spark.network.timeout", "1200s") \
    .config("spark.sql.broadcastTimeout", "1200") \
    .config("spark.python.worker.reuse", False) \
    .getOrCreate()

df = spark.read.csv("datasets/debate-tweets.tsv", header=False, inferSchema=True, sep='\t').toDF(*column_names)


timestamp_col = 'created_at' 


df = df.withColumn('date', to_date(col(timestamp_col), 'EEE MMM dd HH:mm:ss Z yyyy'))


df = df.withColumn("hour", hour(to_timestamp("created_at", "EEE MMM dd HH:mm:ss Z yyyy")))
df = df.withColumn('period', when((col('hour') >= 6) & (col('hour') < 12), 'manhã')
                            .when((col('hour') >= 12) & (col('hour') < 18), 'tarde')
                            .otherwise('noite'))


df = df.withColumn('hashtags', explode(split(lower(col('text')), ' ')).alias('hashtag'))
df = df.filter(col('hashtags').startswith('#'))


print("Contagem total de hashtags após filtrar:", df.count())


df_filtered = df.filter(df.date.between('2014-10-15', '2014-10-20')).cache()

hashtags_period = df_filtered.groupBy('period', 'hashtags').count().orderBy('count', ascending=False)
hashtags_period.show(20)

hashtags_day = df_filtered.groupBy('date', 'hashtags').count().orderBy('count', ascending=False)
hashtags_day.show(20) 

tweets_per_hour = df_filtered.groupBy('date', 'hour').count().orderBy('date', 'hour')
tweets_per_hour.show(20) 


def extract_sentences(text, keyword):
    if text is None:
        return []
    return [sentence for sentence in sent_tokenize(text) if keyword in sentence]

extract_sentences_udf = udf(lambda x: extract_sentences(x, 'Dilma'), ArrayType(StringType()))
df_filtered = df_filtered.withColumn('sentences_dilma', extract_sentences_udf(col('text')))
extract_sentences_udf = udf(lambda x: extract_sentences(x, 'Aécio'), ArrayType(StringType()))
df_filtered = df_filtered.withColumn('sentences_aecio', extract_sentences_udf(col('text')))

sentences_dilma = df_filtered.select(explode(col('sentences_dilma')).alias('sentence')).groupBy('sentence').count().orderBy('count', ascending=False)
sentences_dilma.show(20) 
sentences_aecio = df_filtered.select(explode(col('sentences_aecio')).alias('sentence')).groupBy('sentence').count().orderBy('count', ascending=False)
sentences_aecio.show(20) 
