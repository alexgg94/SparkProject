import pyspark
sc = pyspark.SparkContext()

#Cleaner

import os, shutil
import json
import unicodedata

input = sc.textFile("hdfs:///shared/nando/data/tweets/tweets.json")
#print("Input -> \n" + str(input.take(5)) + "\n")
tweets_lowercased = input.map(lambda x: json.loads(x.lower()))
#print("All -> \n" + str(tweets_lowercased.take(5)) + "\n")
spanish_tweets = tweets_lowercased.filter(lambda t: "es" in t["lang"])
#print("Spanish -> \n" + str(spanish_tweets.collect()) + "\n")
tweets_with_hashtags = spanish_tweets.filter(lambda t: t["entities"]["hashtags"] != [])
#print("Spanish with hashtags -> \n" + str(tweets_with_hashtags.collect()) + "\n")

#TrendingTopics
topics = tweets_with_hashtags.flatMap(lambda t: map(lambda h: (unicodedata.normalize('NFKD', h["text"]).encode('ascii','ignore'),1), t["entities"]["hashtags"]))\
.reduceByKey(lambda a, b: a + b)
print("\nTopics -> " + str(topics.collect()))

#TopNPattern
#if os.path.exists("Results/TopNPattern"): 
#    shutil.rmtree("Results/TopNPattern")

#sc.parallelize(topics.takeOrdered(5, lambda t: -t[1])).saveAsTextFile("Results/TopNPattern")

print(str(topics.takeOrdered(5, lambda t: -t[1])))

#HashtagSentiment
def HashtagSentiment(tweet):
    tweetLength = len(unicodedata.normalize('NFKD', tweet["text"]).encode('ascii','ignore'))
    tweetPolarity = 0.0
    hashtags_with_polarity_and_length = []
    
    for word in unicodedata.normalize('NFKD', tweet["text"]).encode('ascii','ignore').split(" "):
        if len(word) > 0:
            if word[0] == "#":
                tweetPolarity += 0
            elif(word in positive_words):
                tweetPolarity += 1
            elif(word in negative_words):
                tweetPolarity -= 1
            
    for hashtag in tweet["entities"]["hashtags"]:
        hashtags_with_polarity_and_length.append(
        (unicodedata.normalize('NFKD', hashtag["text"]).encode('ascii','ignore'), tweetPolarity/tweetLength))
    return hashtags_with_polarity_and_length

positive_words = set(line.strip().lower() for line in open("/home/gg6/Spark/Word_Classification/positive_words_es.txt"))
negative_words = set(line.strip().lower() for line in open("/home/gg6/Spark/Word_Classification/negative_words_es.txt"))

sentiments_list = tweets_with_hashtags.map(lambda t: HashtagSentiment(t))
flat_sentiments_list = [item for sublist in sentiments_list.collect() for item in sublist]

print("\nHashtag sentiment -> " 
      + str(sc.parallelize(flat_sentiments_list).reduceByKey(lambda a, b: a+b).collect()))
