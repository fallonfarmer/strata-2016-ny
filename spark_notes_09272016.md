# Spark, MLLib, GraphX tutorial

[Git repository](www.github.com/WhiteFangBuck/strata-ny-2016)

## Shell startup
spark-shell --driver-memory 2G --executor-memory 2G --executor-cores 2
* spark shell provides session as "spark", spark context as "sc"
* Spark WebUI: 192.168.16.1:4042


## Linear RegressioLinearRegressionWithEncoding
build pipeline to index and encode categorical variables, create parameters grid

- feature extraction and transformers
- `:pa` paste mode, `Crtl + D` to exit
- `data.show` to view dataframe
- `data.printSchema` to view dtypes
- process:
    1. create dataframe
    2. define pipeline stages, VectorIndexer: specify categorical columns, use string indexer, one hot encoder, vector assembler
    3. instantiate linear regression instance
    4. hypertune with grid
    5. randomly split data into train / test sets
    6. create pipeline using 10 fold cross validation across grid on training data
        - other option: TVS, less computation cost, only one split of data
        - ML Tuning
    7. create holdout from test data using prediction and price columns
    8. evaluate regression with RMSE and R squared

Linear Regression API
- `predictionCol`
- `solver = l-bfgs`
- `tol` tolerance
- regularization defaults to L2
- `setMaxIter`
- `setIntercept` introduce bias

Using grid to hypertune parameters
- `ParamGridBuilder`: elastic net parameters, regularization, fit intercept

## Clustering - K-Means (KMeansExample)
cluster cars data set using k means
use vector assembler to transform data

### Clustering - LDA (TopicModelingExample)
using topic modeling to cluster words, using Clinton email data
repartition data to 20 parts (default is 2)
- speeds up job at task level
- empty partitions are skipped
- if you see disk spills: data structure is too heavy, need more repartitions

data prep
stemming was done beforehand
1. `RegexTokenizer`
2. `StopWordsRemover`
3. `CountVectorizer`

## GraphX
- batch processing

parts of a graph object
- vertex / node = object/person
- edge = relationship
- edge triplet = vertex + edge

create Graph object
- create RDD for vertices (users) and edges (relationships)
- create default user case for missing relationships

Analyze Graph object
- count all postdocs `.filter().count`
- print relationships of users `.triplets.map().collect.foreach(println())`
- `connectComponents` which parts of graph are connected?
- `triangleCount()` how many triangles is the vertex a part of?
- `pageRank()` which has most incoming relationships/edges

`pregel(sc)` function takes vertex, finds the shortest distance
- set source node
- initialize distance between nodes as infinity
- recalculate distances using vertex program
- send messages to nodes: you can apply filters here
- merge message counts

