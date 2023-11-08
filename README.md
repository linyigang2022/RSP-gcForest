# RSP-gcForest: A Distributed Deep Forest via Random Sample Partition.

# preparation

Configure Hadoop and Spark, requiring Spark version 2.4.0. For specific  execution scripts, refer to 'run.sh'. An example script is as  follows.

```sh
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.BLBGCForestSequence \
  --executor-cores 2 \
  --num-executors 16 \
  --driver-memory 4G \
  --executor-memory 16G \
  --conf spark.dynamicAllocation.minExecutors=16 \
  --conf spark.dynamicAllocation.maxExecutors=16 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --dataset watch_acc \
  --train linyigang/data/watch_acc/watch_acc.data \
  --features linyigang/data/watch_acc/features \
  --classNum 18 \
  --casTreeNum 5 \
  --rfNum 1 \
  --crfNum 1 \
  --subRFNum 3 \
  --maxIteration 2 \
  --lambda 0.6
```



