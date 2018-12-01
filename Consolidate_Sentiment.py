import pickle
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
import pyspark
import random

from pyspark import Row
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import functions
from pyspark.sql.functions import col
import pyspark.sql.functions
from pyspark.sql.functions import sum
from pyspark.sql.functions import col, asc, when
import numpy as np

spark = SparkSession.builder.master("local").appName("Yelp").getOrCreate()
sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

from keras.preprocessing import text as txt
tk = txt.Tokenizer(split=" ")

loaded_model = pickle.load(open("SentimentAnalysisModel", 'rb'))
max_length = 500

def predictSentiment(reviewText):
    tk.fit_on_texts(reviewText)
    wordSequences = tk.texts_to_sequences(['',reviewText])
    padded_sequences = sequence.pad_sequences(wordSequences, maxlen = max_length, padding = 'post')
    predictedRating = loaded_model.predict(padded_sequences)
    return float(predictedRating[1][0])


def predictAvgSentiment(reviews):
    predictionList = []
    for review in reviews:
        currPrediction = predictSentiment(review)
        predictionList.append(currPrediction)
    return np.mean(predictionList)


'''review_df = sqlContext.read.json("yelp_academic_dataset_review.json")
business_df = sqlContext.read.json("yelp_academic_dataset_business.json")

business_Reviews = review_df.select("business_id", "text", "stars")
business_df2 = business_df.withColumn("review_count", business_df["review_count"].cast("double"))
business_review_count = business_df2.filter(col('city').isin(['Tempe'])).groupBy("business_id").\
    agg(sum("review_count").alias("rev_count"))
business_gt5 = business_review_count.filter("rev_count >= 5")
right = business_gt5.select("business_id")
tempeBusinessReviews = business_Reviews.join(right, "business_id")'''
# tempeBusinessReviews_1_2_5 = tempeBusinessReviews.filter("stars = 1 or stars = 2 or stars = 5")
# tempeBusinessReviews_Sentiment = tempeBusinessReviews_1_2_5.\
# withColumn("sentiment", when(col("stars") == "1", "0"). when(col("stars") == "2", "0"). when(col("stars") == "5", "1"))


# business_Sentiments.show(5, False)


# business_Sentiments = business_Reviews.select("business_id", "text")
# tempeBusinessReviews.show(5)
# tempeBusinessReviews.coalesce(1).write.csv('tempe_business_reviews.csv')

# business_Sentiments_new = business_Sentiments.rdd.map(tuple).combineByKey(lambda a: [a], lambda a, b: a + [b],
#                                                                           lambda a, b: a + b)
# print(business_Sentiments_new.take(5))


##########################################
# Reading from Tempe Business File below #
##########################################
#
tempe_reviews = sqlContext.read.json("tempe_review_json.json")
tempe_review_sets = tempe_reviews.rdd.map(lambda a : (a[0], str(a[1]))).combineByKey(lambda a: [a], lambda  a, b: a + [b],
                                                                                lambda a, b: a + b)
print(tempe_review_sets.take(5))
# tempe_review_sets_5 = tempe_review_sets.take(5)
tempe_review_sentiments = tempe_review_sets.map(lambda a: (a[0], predictAvgSentiment(a[1])))
tempe_review_sentiments.map(lambda a: Row(a[0], float(a[1]))).toDF().coalesce(1).write.csv("tempe_review_sentiments_new.csv")
print(tempe_review_sentiments.take(5))
