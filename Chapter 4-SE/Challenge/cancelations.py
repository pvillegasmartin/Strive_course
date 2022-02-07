from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('ml-hotels').getOrCreate()

df = spark.read.csv('hotel_bookings.csv', header=True, inferSchema=True)

#Fill na and nulls
df = df.na.fill("")
df = df.na.fill(0)

#Convert string columns to numbers
columns_to_index = ['arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in columns_to_index ]
pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)
df_r=df_r.withColumn('children',df_r['children'].cast("integer").alias('children'))

#Drop columns we wont use
df_r = df_r.drop(*columns_to_index)
df_r = df_r.drop("hotel", "country_index", "company", "agent" ,"reservation_status", "reservation_status_date", "deposit_type_index","customer_type_index")

#Convert all features column to one, spark ML are feed this way
input_cols = list(set(df_r.columns)-set(['is_canceled']))
vectorizer = VectorAssembler(inputCols=input_cols, outputCol='features')
df_r = vectorizer.setHandleInvalid("keep").transform(df_r)

df_train, df_test = df_r.randomSplit([0.8,0.2], seed = 0)
rf_clf = RandomForestClassifier(featuresCol='features', labelCol='is_canceled')
rf_clf = rf_clf.fit(df_train)
df_test = rf_clf.transform(df_test)
df_test = df_test.select('features', 'is_canceled', 'rawPrediction', 'probability', 'prediction')

criterion = MulticlassClassificationEvaluator(labelCol='is_canceled')
acc = criterion.evaluate(df_test)



print(acc)