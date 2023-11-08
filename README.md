# RSP-gcForest: A Distributed Deep Forest via Random Sample Partition.

# preparation

Configure Hadoop and Spark, requiring Spark version 2.4.0. For specific  execution scripts, refer to 'run.sh'. An example script is as  follows.

```sh
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.RSPGCForestSequence \
  --executor-cores 5 \
  --num-executors 4 \
  --driver-memory 30G \
  --executor-memory 32G \
  --conf spark.dynamicAllocation.minExecutors=4 \
  --conf spark.dynamicAllocation.maxExecutors=4 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --dataset [dataset type: covertype, watch_acc, susy or higgs] \
  --train [your train dataset path] \
  --features [features file path, example: see the "/features"] \
  --casTreeNum 20 \
  --rfNum 4 \
  --crfNum 4 \
  --subRFNum 5 \
  --maxIteration 3 \
  --rspBlockNum 240 \
  --g 20
```



