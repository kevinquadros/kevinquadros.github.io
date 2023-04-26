#Import python packages
import os
import pandas as pd
import numpy as np
#Import pyspark packages
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator,␣
,→CrossValidatorModel
from pyspark.ml.feature import Bucketizer, StringIndexer, OneHotEncoder,␣
,→VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator,␣
,→MulticlassClassificationEvaluator
from pyspark.mllib.[Spark Project  (8).pdf](https://github.com/kevinquadros/kevinquadros.github.io/files/11336803/Spark.Project.8.pdf)
evaluation import MulticlassMetrics
#Import visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
# Visualization; Set up visualization parameters
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)
from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})
rcParams['figure.figsize'] = 18,4
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# setting random seed for notebook reproducability
rnd_seed=36
np.random.seed=rnd_seed
np.random.set_state=rnd_seed

#import data
df_train=spark.read.json('gs://kquadr1/Train.json')
df_test=spark.read.json('gs://kquadr1/Test.json')


 Q1: How do the data look like in each dataset? How many
records in each dataset

: df_train.printSchema()
root
|-- _id: string (nullable = true)
|-- arrdelay: double (nullable = true)
|-- carrier: string (nullable = true)
|-- crsarrtime: long (nullable = true)
|-- crsdephour: long (nullable = true)
|-- crsdeptime: long (nullable = true)
|-- crselapsedtime: double (nullable = true)
|-- depdelay: double (nullable = true)
|-- dest: string (nullable = true)
|-- dist: double (nullable = true)
|-- dofW: long (nullable = true)
|-- origin: string (nullable = true)
[4]: df_test.printSchema()
root
|-- _id: string (nullable = true)
|-- arrdelay: double (nullable = true)
|-- carrier: string (nullable = true)
|-- crsarrtime: long (nullable = true)
|-- crsdephour: long (nullable = true)
|-- crsdeptime: long (nullable = true)
|-- crselapsedtime: double (nullable = true)
|-- depdelay: double (nullable = true)
|-- dest: string (nullable = true)
|-- dist: double (nullable = true)
2
|-- dofW: long (nullable = true)
|-- origin: string (nullable = true)
[5]: df_train.select('*').show(5)
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
| _id|arrdelay|carrier|crsarrtime|crsdephour|crsdeptime|crselaps
edtime|depdelay|dest| dist|dofW|origin|
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
|AA_2017-01-01_ATL…| 0.0| AA| 1912| 17| 1700|
132.0| 0.0| LGA| 762.0| 7| ATL|
|AA_2017-01-01_LGA…| 0.0| AA| 1620| 13| 1343|
157.0| 0.0| ATL| 762.0| 7| LGA|
|AA_2017-01-01_MIA…| 10.0| AA| 1137| 9| 939|
118.0| 0.0| ATL| 594.0| 7| MIA|
|AA_2017-01-01_ORD…| 0.0| AA| 26| 20| 2020|
186.0| 0.0| MIA|1197.0| 7| ORD|
|AA_2017-01-01_LGA…| 0.0| AA| 1017| 7| 700|
197.0| 0.0| MIA|1096.0| 7| LGA|
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
only showing top 5 rows
[6]: df_test.select('*').show(5)
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
| _id|arrdelay|carrier|crsarrtime|crsdephour|crsdeptime|crselaps
edtime|depdelay|dest| dist|dofW|origin|
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
|WN_2017-03-01_ATL…| 0.0| WN| 1155| 9| 930|
145.0| 0.0| BOS| 946.0| 3| ATL|
|WN_2017-03-01_ATL…| 210.0| WN| 2215| 19| 1935|
160.0| 243.0| BOS| 946.0| 3| ATL|
|WN_2017-03-01_ATL…| 0.0| WN| 1505| 12| 1235|
150.0| 11.0| BOS| 946.0| 3| ATL|
|WN_2017-03-01_ATL…| 59.0| WN| 1200| 10| 1035|
205.0| 28.0| DEN|1199.0| 3| ATL|
|WN_2017-03-01_ATL…| 29.0| WN| 1450| 13| 1330|
200.0| 16.0| DEN|1199.0| 3| ATL|
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
only showing top 5 rows
3
[7]: df_train.count()
[7]: 41348
[5]: df_test.count()
[5]: 45448

1.1 41348 in the training set and 45448 in the test set
2 Q2: Inspect categorical and numerical variables, what could you
tell from your inspection (5 Points)?
[7]: df_train.describe('origin','crsarrtime').show()
[Stage 6:=============================> (1 + 1) / 2]
+-------+------+------------------+
|summary|origin| crsarrtime|
+-------+------+------------------+
| count| 41348| 41348|
| mean| null|1536.7359485343911|
| stddev| null|496.02225130753914|
| min| ATL| 1|
| max| SFO| 2359|
+-------+------+------------------+
[10]: from pyspark.sql.functions import count, countDistinct
df_train.select(count('*'),count('origin'),countDistinct('origin')).show()
+--------+-------------+----------------------+
|count(1)|count(origin)|count(DISTINCT origin)|
+--------+-------------+----------------------+
| 41348| 41348| 9|
+--------+-------------+----------------------+
[11]: df_test.select('origin').distinct().show()
#the 9 distinct values
+------+
|origin|
+------+
| IAH|
| LGA|
| BOS|
| EWR|
| DEN|
| MIA|
| SFO|
| ATL|
| ORD|
+------+

df_train.groupby('origin').count().show()
# counts of each distinct value
+------+-----+
|origin|count|
+------+-----+
| IAH| 3673|
| LGA| 4992|
| BOS| 3694|
| EWR| 4219|
| DEN| 4269|
| MIA| 4425|
| SFO| 3805|
| ATL| 5971|
| ORD| 6300|
+------+-----+
[8]: df_train.approxQuantile("crsarrtime", \
probabilities=[0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,␣
,→1.0], \
relativeError=0.1)
[8]: [1.0, 1.0, 1.0, 1209.0, 1420.0, 1815.0, 2359.0, 2359.0, 2359.0]
5
2.1 From this inspection we can see that there doesn’t seem to be any missing
data since the counts match. Also for origin there are 9 distinct origin
locations
2.2 For the numeric variable we can see the estimated quantiles, but it can
be debated how useful this is because this data set could be bigger which
would make it more useful
3 4.1 What are the top 5 longest departure delays
[10]: df_train.orderBy(df_train.depdelay.desc()).show(5)
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
| _id|arrdelay|carrier|crsarrtime|crsdephour|crsdeptime|crselaps
edtime|depdelay|dest| dist|dofW|origin|
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
|AA_2017-02-22_SFO…| 1442.0| AA| 1411| 8| 800|
251.0| 1440.0| ORD|1846.0| 3| SFO|
|DL_2017-01-07_BOS…| 1158.0| DL| 2024| 17| 1715|
189.0| 1185.0| ATL| 946.0| 6| BOS|
|UA_2017-02-23_DEN…| 1139.0| UA| 1824| 12| 1244|
220.0| 1138.0| EWR|1605.0| 4| DEN|
|DL_2017-01-22_ORD…| 1090.0| DL| 2240| 19| 1935|
125.0| 1087.0| ATL| 606.0| 7| ORD|
|UA_2017-01-02_MIA…| 1090.0| UA| 2340| 20| 2044|
176.0| 1072.0| EWR|1085.0| 1| MIA|
+--------------------+--------+-------+----------+----------+----------+--------
------+--------+----+------+----+------+
only showing top 5 rows

4 4.2 Average Departure Delay by Carrier
[11]: df_train \
.groupBy("carrier") \
.avg("depdelay") \
.show()
[Stage 11:=============================> (1 + 1) / 2]
+-------+------------------+
|carrier| avg(depdelay)|
+-------+------------------+
| UA|17.477878450696764|
6
| AA| 10.45768118831622|
| DL|15.316061660865241|
| WN|13.491000418585182|
+-------+------------------+
5 4.3 Count of Departure Delays by Carrier (where delay>40 minutes)
[12]: df_train.createOrReplaceTempView("train_flights")
result_df = spark.sql("""
SELECT carrier, count(depdelay)
FROM train_flights
WHERE depdelay >40
GROUP BY carrier
""")
result_df.show()
+-------+---------------+
|carrier|count(depdelay)|
+-------+---------------+
| UA| 2420|
| AA| 757|
| DL| 1043|
| WN| 244|
+-------+---------------+
[ ]:
6 4.4 Count of Departure Delays by Origin
[21]: from pyspark.sql.functions import *
df_train \
.groupBy("origin") \
.agg(count("depdelay")) \
.orderBy(count("depdelay")) \
.show()
+------+---------------+
|origin|count(depdelay)|
+------+---------------+
| IAH| 3673|
| BOS| 3694|
7
| SFO| 3805|
| EWR| 4219|
| DEN| 4269|
| MIA| 4425|
| LGA| 4992|
| ATL| 5971|
| ORD| 6300|
+------+---------------+
7 4.5 Count of Departure Delays by Destination
df_train \
.groupBy("dest") \
.agg(count("depdelay")) \
.orderBy(count("depdelay")) \
.show()
+----+---------------+
|dest|count(depdelay)|
+----+---------------+
| BOS| 3729|
| IAH| 3738|
| SFO| 3818|
| EWR| 4209|
| DEN| 4217|
| MIA| 4439|
| LGA| 4974|
| ATL| 6012|
| ORD| 6212|
+----+---------------+
