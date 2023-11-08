package org.apache.spark.ml.tree.impl

import org.apache.log4j.Logger
import org.apache.parquet.format.IntType

import java.text.SimpleDateFormat
import java.util.Date
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Helper.{UserDefinedFunctions => UDF}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.datasets.{BaseDatasets, Covertype, HIGGS, SUSY, UCI_adult, WatchAcc}
import org.apache.spark.ml.evaluation.{Accuracy, Metric, gcForestEvaluator}
import org.apache.spark.ml.examples.Utils.{TrainParams, trainParser}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, VectorUDT}
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.ml.util.engine.Engine
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, LongType, IntegerType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.SizeEstimator
import org.apache.spark.rdd._

import java.util.concurrent.{Callable, Executors, Future, ThreadPoolExecutor}


private[spark] object RSPGCForestImpl extends Logging {

  def run(
           input: Dataset[_],
           gcforestStategy: GCForestStrategy
         ): RSPGCForestClassificationModel = {
    train(input, strategy = gcforestStategy)
  }

  //  def runWithValidation(
  //                         input: Dataset[_],
  //                         validationInput: Dataset[_],
  //                         gCForestStrategy: GCForestStrategy
  //                       ): RSPGCForestClassificationModel = {
  //    trainWithValidation(input, validationInput, gCForestStrategy)
  //  }

  def runWithValidation(
                         selectedRSP: Array[DataFrame],
                         fullTest: DataFrame,
                         gCForestStrategy: GCForestStrategy
                       ): RSPGCForestClassificationModel = {
    trainWithValidation(selectedRSP, fullTest, gCForestStrategy)
  }

  val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS")

  /**
   * Scan Sequence Data
   *
   * @param dataset    raw input label and features
   * @param windowSize the window size
   * @return
   */

  // create a random forest classifier by type
  def genRFClassifier(rfType: String,
                      strategy: GCForestStrategy,
                      isScan: Boolean,
                      num: Int): GCForestClassifier = {
    val rf = rfType match {
      case "rfc" => new GCForestClassifier().setRFNum(strategy.subRFNum).setCRFNum(0).setMaxIteration(1)
      case "crfc" => new GCForestClassifier().setRFNum(0).setCRFNum(strategy.subRFNum).setMaxIteration(1)
    }
    val gcForest = rf
      .setDataSize(strategy.dataSize)
      .setDataStyle(strategy.dataStyle)
      .setMultiScanWindow(strategy.multiScanWindow)
      .setCascadeForestTreeNum(strategy.cascadeForestTreeNum)
      .setScanForestTreeNum(strategy.scanForestTreeNum)
      .setMaxDepth(strategy.maxDepth)
      .setMaxBins(strategy.maxBins)
      .setMinInfoGain(strategy.minInfoGain)
      .setMaxMemoryInMB(strategy.maxMemoryInMB)
      .setCacheNodeId(strategy.cacheNodeId)
      .setScanForestMinInstancesPerNode(strategy.scanMinInsPerNode)
      .setCascadeForestMinInstancesPerNode(strategy.cascadeMinInsPerNode)
      .setFeatureSubsetStrategy(strategy.featureSubsetStrategy)
      .setCrf_featureSubsetStrategy(strategy.crf_featureSubsetStrategy)
      .setEarlyStoppingRounds(strategy.earlyStoppingRounds)
      .setIDebug(strategy.idebug)
      .setNumClasses(strategy.classNum)

    gcForest
  }

  /**
   * Concat multi-scan features
   *
   * @param dataset one of a window
   * @param sets    the others
   * @return input for Cascade Forest
   */
  def concatenate(
                   strategy: GCForestStrategy,
                   dataset: Dataset[_],
                   sets: Dataset[_]*
                 ): DataFrame = {
    val sparkSession = dataset.sparkSession
    var unionSet = dataset.toDF()
    sets.foreach(ds => unionSet = unionSet.union(ds.toDF()))

    class Record(val instance: Long, // instance id
                 val label: Double, // label
                 val features: Vector, // features
                 val scanId: Int, // the scan id for multi-scan
                 val forestId: Int, // forest id
                 val winId: Long) extends Serializable // window id

    val concatData = unionSet.select(
      strategy.instanceCol, strategy.labelCol,
      strategy.featuresCol, strategy.scanCol,
      strategy.forestIdCol, strategy.winCol).rdd.map {
      row =>
        val instance = row.getAs[Long](strategy.instanceCol)
        val label = row.getAs[Double](strategy.labelCol)
        val features = row.getAs[Vector](strategy.featuresCol)
        val scanId = row.getAs[Int](strategy.scanCol)
        val forestId = row.getAs[Int](strategy.forestIdCol)
        val winId = row.getAs[Long](strategy.winCol)

        new Record(instance, label, features, scanId, forestId, winId)
    }.groupBy(
      record => record.instance
    ).map { group =>
      val instance = group._1
      val records = group._2
      val label = records.head.label

      def recordCompare(left: Record, right: Record): Boolean = {
        var code = left.scanId.compareTo(right.scanId)
        if (code == 0) code = left.forestId.compareTo(right.forestId)
        if (code == 0) code = left.winId.compareTo(right.winId)
        code < 0
      }

      val features = new DenseVector(records.toSeq.sortWith(recordCompare)
        .flatMap(_.features.toArray).toArray)
      // features = [0, 0, ..., 0] (903 dim)
      Row.fromSeq(Array[Any](instance, label, features))
    }

    val schema: StructType = StructType(Seq[StructField]())
      .add(StructField(strategy.instanceCol, LongType))
      .add(StructField(strategy.labelCol, DoubleType))
      .add(StructField(strategy.featuresCol, new VectorUDT))
    sparkSession.createDataFrame(concatData, schema)
  }

  /**
   * concat inputs of Cascade Forest with prediction
   *
   * @param feature input features
   * @param predict prediction features
   * @return
   */
  def mergeFeatureAndPredict(
                              feature: Dataset[_],
                              predict: Dataset[_],
                              strategy: GCForestStrategy): DataFrame = {
    val vectorMerge = udf { (v1: Vector, v2: Vector) =>
      new DenseVector(v1.toArray.union(v2.toArray))
    }

    if (predict != null) {
      feature.join(
        // join (predict feature col to predictionCol)
        predict.withColumnRenamed(strategy.featuresCol, strategy.predictionCol),
        Seq(strategy.instanceCol, "rsp") // join on instanceCol
        // add a featureCol with featureCol + predictionCol
      ).withColumn(strategy.featuresCol, vectorMerge(col(strategy.featuresCol),
        col(strategy.predictionCol))
      ).select(strategy.instanceCol, strategy.featuresCol, strategy.labelCol, "rsp").toDF()
      // select 3 cols to DataFrame
    } else {
      feature.toDF()
    }
  }

  private def getNowTime = dateFormat.format(new Date())

  /**
   * Multi-Grained Scanning
   */
  def multi_grain_Scan(
                        dataset: Dataset[_],
                        strategy: GCForestStrategy): DataFrame = {

    require(dataset != null, "Null dataset need not to scan")
    // scalastyle:off println
    var scanFeature: DataFrame = null
    val rand = new Random()
    rand.setSeed(System.currentTimeMillis())

    //    logInfo(s"[$getNowTime] Multi Grained Scanning begin!")

    if (strategy.dataStyle == "Seq") {
      scanFeature = dataset.toDF()
    }
    // scanFeature: (instanceId, label, features)
    //    logInfo(s"[$getNowTime] Multi Grained Scanning finished!")
    // scalastyle:on println
    scanFeature
  }

  /**
   * train a RSPGCForestClassificationModel without validation
   *
   * @param input
   * @param strategy
   * @return
   */
  def train(
             input: Dataset[_],
             strategy: GCForestStrategy): RSPGCForestClassificationModel = {
    val numClasses: Int = strategy.classNum
    val erfModels = ArrayBuffer[Array[GCForestClassificationModel]]()
    new RSPGCForestClassificationModel(erfModels.toArray, numClasses)
  }

  private def reLoadData(strategy: GCForestStrategy, mergeData: (Dataset[Row], Dataset[Row])) = {
    val sparkSession = mergeData._1.sparkSession
    val vector2Array = udf { (v1: Vector) =>
      v1.toArray.mkString(",")
    }
    val randomDouble = Random.nextDouble()
    val trainFile = s"linyigang/data/temp/trainFile${randomDouble}.data"
    val testFile = s"linyigang/data/temp/testFile${randomDouble}.data"

    println(s"trainFile: ${trainFile}")
    println(s"testFile: ${testFile}")
    val featuresFile = strategy.newFeatures
    mergeData._1.withColumn(strategy.featuresCol, vector2Array(col(strategy.featuresCol))).repartition(1).write.
      mode("overwrite").option("quote", " ").option("header", "false").csv(trainFile)
    mergeData._2.withColumn(strategy.featuresCol, vector2Array(col(strategy.featuresCol))).repartition(1).write.
      mode("overwrite").option("quote", " ").option("header", "false").csv(testFile)
    val (layerTraining, layerTesting) = {
      val train = new BaseDatasets().load_data1(sparkSession, trainFile, sparkSession.sparkContext.defaultParallelism)
      val test = new BaseDatasets().load_data1(sparkSession, testFile, sparkSession.sparkContext.defaultParallelism)
      (train, test)
    }
    (layerTraining, layerTesting)
  }

  /**
   * Train a Cascade Forest with validation
   */
  private def trainWithValidation(
                                   selectedRSP: Array[DataFrame],
                                   fullTest: DataFrame,
                                   strategy: GCForestStrategy): RSPGCForestClassificationModel = {
    val logger = Logger.getLogger(this.getClass)
    val timer = new TimeTracker()
    timer.start("total")


    // multi grain scan. For structured data, multi-granularity scanning will return directly.
    val numClasses: Int = strategy.classNum
    // TODO: better representation to model
    val erfModels = ArrayBuffer[Array[GCForestClassificationModel]]()
    val g = selectedRSP.length

    var unionTrain: DataFrame = null
    for (i <- 0 until g) {
      val rsp = selectedRSP(i).withColumn("rsp", lit(i.toLong))
      unionTrain = if (unionTrain == null) rsp else unionTrain.union(rsp)
    }
    val unionTest = fullTest.withColumn("rsp", lit(0.toLong))
    //    unionTrain.printSchema()
    //    unionTest.printSchema()

    val scanFeatureTrain = multi_grain_Scan(unionTrain, strategy).cache()
    val scanFeatureTest = multi_grain_Scan(unionTest, strategy).cache()

    //    val scanFeature = Array.ofDim[(DataFrame, DataFrame)](g)
    //    for (i <- 0 until g) {
    //      scanFeature(i) = (multi_grain_Scan(selectedRSP(i)._1, strategy), multi_grain_Scan(selectedRSP(i)._2, strategy))
    //      scanFeature(i)._1.cache
    //      scanFeature(i)._2.cache
    //    }

    //    val scanFeatureTrain: DataFrame = null
    //    scanFeatureTrain = (multi_grain_Scan(unionTrain, strategy), multi_grain_Scan(unionTest, strategy))
    //    val scanFeatureTest: DataFrame = null
    //    scanFeatureTest = (multi_grain_Scan(unionTrain, strategy), multi_grain_Scan(unionTest, strategy))

    logInfo(s"scanFeature finished")

    timer.start("init")

    // Init
    val sparkSession = scanFeatureTrain.sparkSession
    val sc = sparkSession.sparkContext
    val rng = new Random()
    rng.setSeed(System.currentTimeMillis())

    var lastPredictionTrain: DataFrame = null
    var lastPredictionTest: DataFrame = null
    var ensemblePredictTrain: DataFrame = null
    var ensemblePredictTest: DataFrame = null

    val acc_list = Array(ArrayBuffer[Double](), ArrayBuffer[Double]())

    var layer_train_metric: Accuracy = new Accuracy(0, 0) // closure need
    var layer_test_metric: Accuracy = new Accuracy(0, 0) // closure need

    val maxIteration = strategy.maxIteration
    require(maxIteration > 0, "Non-positive maxIteration")
    var layer_id = 1
    var reachMaxLayer = false

    val executors = Executors.newFixedThreadPool(strategy.rfNum + strategy.crfNum).asInstanceOf[ThreadPoolExecutor]
    timer.stop("init")

    // start to train the layer ${layer_id}
    while (!reachMaxLayer) {
      import java.lang.Runtime

      val runtime = Runtime.getRuntime
      println("Total Memory: " + runtime.totalMemory)
      println("Free Memory: " + runtime.freeMemory)
      println("Used Memory: " + (runtime.totalMemory - runtime.freeMemory))

      logger.info(s"[$getNowTime] Training rsp Forest Layer $layer_id")

      // Create GCForestClassifier based on random forest type
      val gcForests = (
        Range(0, strategy.rfNum).map(it =>
          genRFClassifier("rfc", strategy, isScan = false, num = rng.nextInt + it))
          ++
          Range(strategy.rfNum, strategy.rfNum + strategy.crfNum).map(it =>
            genRFClassifier("crfc", strategy, isScan = false, num = rng.nextInt + it))
        ).toArray[GCForestClassifier]
      assert(gcForests.length == strategy.rfNum + strategy.crfNum, "rsp random Forests inValid!")

      // scanFeatures_*: (instanceId, label, features)
      logInfo(s"merge scanFeature and lastPrediction...")
      timer.start("merge scanFeature and lastPrediction")

      //      var mergedData = Array.ofDim[(Dataset[Row], Dataset[Row])](g)
      //      logInfo("start to mergeFeatureAndPredict")
      //      val mergeExecutors = Executors.newFixedThreadPool(g).asInstanceOf[ThreadPoolExecutor]
      //      var mergeTasks = Array.ofDim[Future[(DataFrame, DataFrame)]](g)
      //      for (i <- 0 until g) {
      //        val mergeTask = mergeExecutors.submit(new MergeTask(strategy, scanFeature(i), lastPrediction(i)))
      //        mergeTasks(i) = mergeTask
      //      }
      //      for (i <- 0 until g) {
      //        mergedData(i) = mergeTasks(i).get()
      //        logInfo(s"mergedData($i) finished")
      //      }
      //      mergeExecutors.shutdown()

      //      for (i <- 0 until strategy.rfNum + strategy.crfNum; j <- 0 until strategy.subRFNum) {
      //        mergedData(i)(j) = (mergeFeatureAndPredict(scanFeature(i)(j)._1, lastPrediction(i)(j)._1, strategy).coalesce(sc.defaultParallelism).cache,
      //          mergeFeatureAndPredict(scanFeature(i)(j)._2, lastPrediction(i)(j)._2, strategy).coalesce(sc.defaultParallelism).cache)
      //        mergedData(i)(j)._1.count()
      //        mergedData(i)(j)._2.count()
      //      }


      var training = mergeFeatureAndPredict(scanFeatureTrain, lastPredictionTrain, strategy)
        .coalesce(sc.defaultParallelism)
      var testing = mergeFeatureAndPredict(scanFeatureTest, lastPredictionTest, strategy)
        .coalesce(sc.defaultParallelism)


      timer.stop("merge scanFeature and lastPrediction")

      logInfo("reload merged data")
      timer.start("reload merged data")

      if (layer_id >= 2) {
        val vector2Array = udf { (v1: Vector) =>
          v1.toArray.mkString(",")
        }
        val randomDouble = Random.nextDouble()
        val trainFile = s"linyigang/data/temp/trainFile${randomDouble}.data"
        val testFile = s"linyigang/data/temp/testFile${randomDouble}.data"
        //    println(s"trainFile: ${trainFile}")
        //    println(s"testFile: ${testFile}")
        training.withColumn(strategy.featuresCol, vector2Array(col(strategy.featuresCol))).repartition(1).write.
          mode("overwrite").option("quote", " ").option("header", "false").csv(trainFile)
        testing.withColumn(strategy.featuresCol, vector2Array(col(strategy.featuresCol))).repartition(1).write.
          mode("overwrite").option("quote", " ").option("header", "false").csv(testFile)

        training = new BaseDatasets().load_data2(sparkSession, trainFile, sparkSession.sparkContext.defaultParallelism).coalesce(sparkSession.sparkContext.defaultParallelism).cache()
        testing = new BaseDatasets().load_data2(sparkSession, testFile, sparkSession.sparkContext.defaultParallelism).coalesce(sparkSession.sparkContext.defaultParallelism).cache()
        training.count
        testing.count()
      }else{
        training.cache()
        testing.cache()
        training.count
        testing.count()
      }

      //      val reloadExecutors = Executors.newFixedThreadPool(g).asInstanceOf[ThreadPoolExecutor]
      //      var reloadTasks = Array.ofDim[Future[(DataFrame, DataFrame)]](g)
      //      if (layer_id >= 2) {
      //        for (i <- 0 until g) {
      //          val reloadTask = reloadExecutors.submit(new ReloadTask(strategy, mergedData(i)))
      //          reloadTasks(i) = reloadTask
      //        }
      //        for (i <- 0 until g) {
      //          mergedData(i) = reloadTasks(i).get()
      //        }
      //      }
      //      reloadExecutors.shutdown()
      //            if (layer_id >= 2) {
      //              for (i <- 0 until g) {
      //                val vector2Array = udf { (v1: Vector) =>
      //                  v1.toArray.mkString(",")
      //                }
      //                val randomDouble = Random.nextDouble()
      //                val trainFile = s"linyigang/data/temp/trainFile${randomDouble}.data"
      //                val testFile = s"linyigang/data/temp/testFile${randomDouble}.data"
      //                //    println(s"trainFile: ${trainFile}")
      //                //    println(s"testFile: ${testFile}")
      //                mergedData(i)._1.withColumn(strategy.featuresCol, vector2Array(col(strategy.featuresCol))).repartition(1).write.
      //                  mode("overwrite").option("quote", " ").option("header", "false").csv(trainFile)
      //                mergedData(i)._2.withColumn(strategy.featuresCol, vector2Array(col(strategy.featuresCol))).repartition(1).write.
      //                  mode("overwrite").option("quote", " ").option("header", "false").csv(testFile)
      //                mergedData(i)._1.unpersist()
      //                mergedData(i)._2.unpersist()
      //
      //                mergedData(i) = {
      //                  val train = new BaseDatasets().load_data1(sparkSession, trainFile, sparkSession.sparkContext.defaultParallelism).coalesce(sparkSession.sparkContext.defaultParallelism).cache()
      //                  val test = new BaseDatasets().load_data1(sparkSession, testFile, sparkSession.sparkContext.defaultParallelism).coalesce(sparkSession.sparkContext.defaultParallelism).cache()
      //                  (train, test)
      //                }
      //                logInfo(s"reload mergedata ${i}")
      //              }
      //            }
      timer.stop("reload merged data")

      logInfo("random choose (rfNum*subRFNum) RSP blocks")
      timer.start("random choose (rfNum*subRFNum) RSP blocks")
      //      val features_dim = mergedData(0)._1.first().mkString.split(",").length // action, get training truly
      //
      //      logger.info(s"[$getNowTime] Training Set = ($n_train, $features_dim), " + s"Testing Set = ($n_test, $features_dim)")


      ensemblePredictTrain = null
      ensemblePredictTest = null

      // rsp sampling, sample the train set and test set
      val r = strategy.subRFNum
      val sampleTrain = Array.ofDim[DataFrame](strategy.rfNum + strategy.crfNum, strategy.subRFNum)
      val selectedIndexesR = Random.shuffle(0 to g - 1).take(r).toArray
      val sampleTrainCount = Array.ofDim[Int](strategy.rfNum + strategy.crfNum)
      for (i <- 0 to (strategy.rfNum + strategy.crfNum) - 1) {
        val selectedIndexesR = Random.shuffle(0 to g - 1).take(r).toArray
        println("selectedIndexes inner: " + selectedIndexesR.mkString(","))
        //        training.printSchema()
        var count = 0
        for (j <- 0 to r - 1) {
          sampleTrain(i)(j) = training.filter(col("rsp") === selectedIndexesR(j))
          sampleTrain(i)(j).cache()
          sampleTrain(i)(j).count()
          count += sampleTrain(i)(j).count().toInt
          //          logInfo(s"selected ${sampleMergedData(i)(j)._1.count()} training")
          //          logInfo(s"selected ${sampleMergedData(i)(j)._2.count()} testing")
        }
        sampleTrainCount(i) = count
      }
      println(s"sampleTrainCount.mkString:${sampleTrainCount.mkString(",")}")
      val sampleTrainPer = Array.ofDim[Double](strategy.rfNum + strategy.crfNum)
      for (i <- 0 to (strategy.rfNum + strategy.crfNum) - 1) {
        sampleTrainPer(i) = sampleTrainCount(i).toDouble / sampleTrainCount.sum
      }
      println(s"sampleTrainPer.mkString:${sampleTrainPer.mkString(",")}")

      layer_train_metric.reset()
      layer_test_metric.reset()

      logger.info(s"rsp sub gcforest, the size of train set: ${training.count()}")
      logger.info(s"rsp random forest num：${strategy.rfNum}")
      logger.info(s"rsp completely random forest num：${strategy.crfNum}")
      logger.info(s"rsp sub random forest num：${strategy.subRFNum}")
      logger.info(s"start to cache mergedData......")


      timer.stop("random choose (rfNum*subRFNum) RSP blocks")

      logger.info(s"randomForests training...")
      timer.start("randomForests training")

      //-----------------------------------------------------------
      // Train the gcforest, the transformation process is completed during the training process.
      // It has high speed but high communication cost.

      val stime = System.currentTimeMillis()
      var list = Array[Future[(Dataset[Row], Dataset[Row], GCForestClassificationModel)]]()
      logInfo(s"rsp layer [${layer_id}] gcForests subforest fitting and transforming ......")

      gcForests.zipWithIndex.foreach { case (gcforest, it) =>
        if (strategy.idebug) println(s"[$getNowTime] timer.start(cvClassVectorGeneration)")

        //        val model = gcforest.train(sampleTraining(it), sampleTesting(it))
        val task = executors.submit(
          new RSPGCForestTask4(sparkSession, gcforest, training, testing, sampleTrain, it)
        )
        list :+= task
      }

      //      logger.info(s"The tasks have been submitted")
      val transformedResults = list.map(result => {
        val transformed = result.get()
        //        println(model.toString)
        transformed
      })
      val totalTime = (System.currentTimeMillis() - stime) / 1000.0
      logInfo(s"Total time for all GCForest training: $totalTime, 开始最后转换")
      erfModels += transformedResults.map {
        _._3
      }.toArray
      timer.stop("randomForests training")

      logInfo("aaa")
      logInfo("get ensemblePrediction sequentially")
      timer.start("get ensemblePrediction sequentially")
      transformedResults.zipWithIndex.foreach { case (transformed, it) =>
        //        logInfo(s"transformedxx: ${transformed._1(0)._1.first().asInstanceOf[Row].mkString}")


        val predict = transformed._1.drop(strategy.featuresCol)
          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
          .withColumn(strategy.forestIdCol, lit(it))
          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol, "rsp")

        val predict_test = transformed._2.drop(strategy.featuresCol)
          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
          .withColumn(strategy.forestIdCol, lit(it))
          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol, "rsp")

        ensemblePredictTrain = if (ensemblePredictTrain == null) predict else ensemblePredictTrain.union(predict)
        ensemblePredictTest = if (ensemblePredictTest == null) predict_test else ensemblePredictTest.union(predict_test)

        //          logInfo(s"ensemblePredict($i)._1.count():${ensemblePredict(i)._1.count()}")
        //          logInfo(s"ensemblePredictxx: ${ensemblePredict(i)._1.first().asInstanceOf[Row].mkString}")

        //          val train_result = transformed._1(i)._1
        //            .drop(strategy.featuresCol)
        //            .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        //          val test_result = transformed._1(i)._2
        //            .drop(strategy.featuresCol)
        //            .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        //          val train_acc = gcForestEvaluator.evaluateAccuracyPartition(train_result)
        //          val test_acc = gcForestEvaluator.evaluateAccuracyPartition(test_result)
        //          layer_train_metric = layer_train_metric + train_acc
        //          layer_test_metric = layer_test_metric + test_acc

        logDebug(s"get ensemblePrediction ${it} finished")


        //-----------------------------------------------------------
        //      // Train the gcforest, and transform the dataset. It has low communication cost but slow speed.
        //      var list = Array[Future[GCForestClassificationModel]]()
        //
        //      gcForests.zipWithIndex.foreach { case (gcforest, it) =>
        //        println(s"[$getNowTime] blb layer [${layer_id}] gcForests subforest [${it}] fitting and transforming ......")
        //        if (strategy.idebug) println(s"[$getNowTime] timer.start(cvClassVectorGeneration)")
        //
        //        //        val model = gcforest.train(sampleTraining(it), sampleTesting(it))
        //        val task = executors.submit(
        //          new RSPGCForestTask3(sparkSession, gcforest, mergedData(it))
        //        )
        //        list :+= task
        //      }
        //      val totalTime = (System.currentTimeMillis() - stime) / 1000.0
        //      logger.info(s"The task has been submitted")
        //      val erfModel = list.map(result => {
        //        val model = result.get()
        ////        println(model.toString)
        //        model
        //      })
        //
        //      println(s"Total time for all GCForest training: $totalTime, 开始最后转换")
        //      erfModels += erfModel
        //      var transformed = Array.ofDim[(Dataset[Row], Dataset[Row])](strategy.rfNum + strategy.crfNum, strategy.subRFNum)
        //
        //
        //      erfModel.zipWithIndex.foreach { case (model, it) =>
        //
        //        for (i <- 0 until strategy.rfNum + strategy.crfNum; j <- 0 until strategy.subRFNum) {
        //          transformed(i)(j) = (model.transform(mergedData(i)(j)._1),
        //            model.transform(mergedData(i)(j)._2))
        //
        //          val predict = transformed(i)(j)._1
        //            .withColumn(strategy.forestIdCol, lit(it))
        //            .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)
        //
        //          val predict_test = transformed(i)(j)._2
        //            .withColumn(strategy.forestIdCol, lit(it))
        //            .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)
        //
        //          val ep = if (ensemblePredict(i)(j) == null) predict else ensemblePredict(i)(j)._1.union(predict)
        //          val ept = if (ensemblePredict(i)(j) == null) predict_test else ensemblePredict(i)(j)._2.union(predict_test)
        //
        //          ensemblePredict(i)(j) = (ep, ept)
        //          //          logInfo(s"ensemblePredict(i)(j)._1.count():${ensemblePredict(i)(j)._1.count()}")
        //
        //          val train_result = transformed(i)(j)._1
        //            .drop(strategy.featuresCol)
        //            .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        //          val test_result = transformed(i)(j)._2
        //            .drop(strategy.featuresCol)
        //            .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        //          val train_acc = gcForestEvaluator.evaluateAccuracyPartition(train_result)
        //          val test_acc = gcForestEvaluator.evaluateAccuracyPartition(test_result)
        //          layer_train_metric = layer_train_metric + train_acc
        //          layer_test_metric = layer_test_metric + test_acc
        //
        //        }

        //        for (i <- 0 until strategy.rfNum + strategy.crfNum; j <- 0 until strategy.subRFNum) {
        //          mergedData(i)(j) = (mergeFeatureAndPredict(scanFeature(i)(j)._1, lastPrediction(i)(j)._1, strategy).coalesce(sc.defaultParallelism),
        //            mergeFeatureAndPredict(scanFeature(i)(j)._2, lastPrediction(i)(j)._2, strategy).coalesce(sc.defaultParallelism))
        //        }
        //
        //        val transformedTrain = model.transform(training).drop(strategy.featuresCol)
        //          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        //        val transformedTest = model.transform(testing).drop(strategy.featuresCol)
        //          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        //        val predict = transformedTrain
        //          .withColumn(strategy.forestIdCol, lit(it))
        //          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)
        //
        //        val predict_test = transformedTest
        //          .withColumn(strategy.forestIdCol, lit(it))
        //          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)
        //        ensemblePredict =
        //          if (ensemblePredict == null) predict else ensemblePredict.union(predict)
        //        ensemblePredict_test =
        //          if (ensemblePredict_test == null) predict_test else ensemblePredict_test.union(predict_test)
        //
        //        val train_result = transformedTrain
        //          .drop(strategy.featuresCol)
        //          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        //        val test_result = transformedTest
        //          .drop(strategy.featuresCol)
        //          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        //        val train_acc = gcForestEvaluator.evaluateAccuracyPartition(train_result)
        //        val test_acc = gcForestEvaluator.evaluateAccuracyPartition(test_result)
        //        layer_train_metric = layer_train_metric + train_acc
        //        layer_test_metric = layer_test_metric + test_acc
      }
      timer.stop("get ensemblePrediction sequentially")
      //-----------------------------------------------------------


      //      logger.info(s"[$getNowTime] Forests fitting and transforming finished!")

      acc_list(0) += layer_train_metric.getAccuracy
      acc_list(1) += layer_test_metric.getAccuracy

      logInfo(s"get unstacked prediction ......")
      timer.start("flatten prediction(get unstacked prediction, stack prediction, cache prediction)")
      timer.start("get unstacked prediction")

      //      val getPredictionExecutors = Executors.newFixedThreadPool(g).asInstanceOf[ThreadPoolExecutor]
      //      var getPredictionTasks = Array.ofDim[Future[Array[RDD[Row]]]](g)
      //      //      logInfo(s"ensemblePredictxx: ${ensemblePredict(0)._1.first().asInstanceOf[Row].mkString}")
      //      for (i <- 0 until g) {
      //        val getPredictionTask = getPredictionExecutors.submit(new GetPredictionTask(sparkSession, ensemblePredict(i), strategy))
      //        getPredictionTasks(i) = getPredictionTask
      //      }
      //      logInfo(s"submitted")
      //      for (i <- 0 until g) {
      //        predictRDDs(i) = getPredictionTasks(i).get()
      //        logInfo(s"getting predictRDDs($i) finished")
      //      }
      //      getPredictionExecutors.shutdown()

      val predictRDDs =
        Array(ensemblePredictTrain, ensemblePredictTest).map { predict =>
          val grouped = predict.rdd.groupBy(row => Seq(row.getAs[Long](strategy.instanceCol), row.getAs[Long]("rsp")))
          val predictRDD = grouped.map { group =>
            val instanceId = group._1(0)
            val rsp = group._1(1)
            val rows = group._2
            val features = new DenseVector(rows.toArray
              .sortWith(_.getAs[Int](strategy.forestIdCol) < _.getAs[Int](strategy.forestIdCol))
              .flatMap(_.getAs[Vector](strategy.featuresCol).toArray))
            Row.fromSeq(Array[Any](instanceId, rsp, features))
          }
          predictRDD
        }
      //        logInfo(s"predictRDDs(i)(j)(0).count():${predictRDDs(i)(j)(0).count()}")
      logInfo(s"getting predictRDDs finished")
      timer.stop("get unstacked prediction")
      logInfo("stack prediction.")
      timer.start("stack prediction.")
      val schema = new StructType()
        .add(StructField(strategy.instanceCol, LongType))
        .add(StructField("rsp", LongType))
        .add(StructField(strategy.featuresCol, new VectorUDT))
      val vectorMerge = udf { (v1: Vector) =>
        val avgPredict = Array.fill[Double](numClasses)(0d)
        val lastPredict = v1.toArray
        lastPredict.indices.foreach { i =>
          val classType = i % numClasses
          avgPredict(classType) = avgPredict(classType) + lastPredict(i)
        }
        val n = lastPredict.length / numClasses

        new DenseVector(avgPredict.map(predict => {
          predict / n
        }))
      }

      // unpersist lastPrediction
      if (layer_id >= 2) {
        lastPredictionTrain.unpersist()
        lastPredictionTest.unpersist()
      }
      //        println(predictRDDs(i)(j))
      lastPredictionTrain = sparkSession.createDataFrame(predictRDDs(0), schema)
        .coalesce(sc.defaultParallelism)
        .withColumn(strategy.featuresCol, vectorMerge(col(strategy.featuresCol))).cache()
      lastPredictionTest = sparkSession.createDataFrame(predictRDDs(1), schema)
        .coalesce(sc.defaultParallelism)
        .withColumn(strategy.featuresCol, vectorMerge(col(strategy.featuresCol))).cache()
      timer.stop("stack prediction.")

      logInfo("cache last prediction")
      timer.start("cache last prediction")
      lastPredictionTrain.count()
      lastPredictionTest.count()
      logInfo(s"after stack, lastPrediction dim：${lastPredictionTrain.first().mkString.split(",").length}")
      //      val cacheExecutors = Executors.newFixedThreadPool(g).asInstanceOf[ThreadPoolExecutor]
      //      var cacheTasks = Array.ofDim[Future[(DataFrame, DataFrame)]](g)
      //      for (i <- 0 until g) {
      //        val cacheTask = cacheExecutors.submit(new CacheTask(lastPrediction(i)))
      //        cacheTasks(i) = cacheTask
      //      }
      //      for (i <- 0 until g) {
      //        lastPrediction(i) = cacheTasks(i).get()
      //        logInfo(s"cache lastPrediction($i) finished")
      //      }
      //      cacheExecutors.shutdown()
      //      for (i <- 0 until g) {
      //        lastPrediction(i)._1.cache()
      //        lastPrediction(i)._2.cache()
      //        lastPrediction(i)._1.count()
      //        lastPrediction(i)._2.count()
      //        logInfo(s"cache lastPrediction($i) finished")
      //      }
      timer.stop("cache last prediction")

      //      logDebug(s"RSP-gcForest lastPrediction:(${lastPrediction(0)._1.count},${lastPrediction(0)._1.first().asInstanceOf[Row].mkString.split(",").length})")
      //      logDebug(s"RSP-gcForest lastPrediction_test:(${lastPrediction(0)._2.count},${lastPrediction(0)._2.first().asInstanceOf[Row].mkString.split(",").length})")

      //      logInfo(s"lastPrediction: ${lastPrediction(0)._1.first().asInstanceOf[Row].mkString}")
      //      logInfo(s"sampleMergedData: ${sampleMergedData(0)(0)._1.first().asInstanceOf[Row].mkString}")
      // get predictRDDs, [instanceId, [features1, ..., features${rfNum}]]      // feature = probability
      //      val predictRDDs =
      //        Array(ensemblePredict, ensemblePredict_test).map { predict =>
      //          val grouped = predict.rdd.groupBy(_.getAs[Long](strategy.instanceCol))
      //          val predictRDD = grouped.map { group =>
      //            val instanceId = group._1
      //            val rows = group._2
      //            val features = new DenseVector(rows.toArray
      //              .sortWith(_.getAs[Int](strategy.forestIdCol) < _.getAs[Int](strategy.forestIdCol))
      //              .flatMap(_.getAs[Vector](strategy.featuresCol).toArray))
      //            Row.fromSeq(Array[Any](instanceId, features))
      //          }
      //          predictRDD
      //        }
      //      val predictRDDDim = predictRDDs(0).first().mkString.split(",").length

      // Update the last prediction, and it will be concatenated with the raw feature to become the new feature of the next layer.
      //      val schema = new StructType()
      //        .add(StructField(strategy.instanceCol, LongType))
      //        .add(StructField(strategy.featuresCol, new VectorUDT))
      //      lastPrediction(i)(j) = (sparkSession.createDataFrame(predictRDDs(0), schema)
      //        .coalesce(sc.defaultParallelism),
      //        sparkSession.createDataFrame(predictRDDs(1), schema)
      //          .coalesce(sc.defaultParallelism))

      // modify the probability Aggregation strategy from concat mode to stack mode

      //      val predictRDDDim = predictRDDs(0)(0).first().mkString.split(",").length
      //      logInfo(s"before stack, lastPrediction dim: ${predictRDDDim}. after stack, lastPrediction dim：${lastPrediction(0)._1.first().mkString.split(",").length}")

      //      logDebug(s"[$getNowTime] rsp gcforestImpl layer train finish, predict rdd feature dim = ($predictRDDDim)")
      timer.stop("flatten prediction(get unstacked prediction, stack prediction, cache prediction)")
      logInfo(s"Get prediction RDD finished! Layer $layer_id training finished!")

      val opt_layer_id_train = acc_list(0).zipWithIndex.maxBy(_._1)._2
      val opt_layer_id_test = acc_list(1).zipWithIndex.maxBy(_._1)._2

      // early stop


      logInfo(s"[$getNowTime] rsp layer [${layer_id}] gcForests summary" +
        s"[Result]  Layer] layer_num = $layer_id " +
        "accuracy_train=%.3f%%, ".format(layer_train_metric.getAccuracy * 100) +
        "accuracy_test=%.3f%%".format(layer_test_metric.getAccuracy * 100))

      logger.info(s"RSP-GCForestImpl layer:${layer_id} time cost:")
      logger.info(s"$timer")


      // early stop, if there is no improvement in accuracy for "earlyStoppingRounds".
      val outOfRounds = (strategy.earlyStopByTest && layer_id - opt_layer_id_test >= strategy.earlyStoppingRounds) ||
        (!strategy.earlyStopByTest && layer_id - opt_layer_id_train >= strategy.earlyStoppingRounds)
      if (outOfRounds) {
        logger.info(s"[$getNowTime] " +
          s"[Result][Optimal Level Detected] opt_layer_id = " +
          s"${if (strategy.earlyStopByTest) opt_layer_id_test else opt_layer_id_train}, " +
          "accuracy_train=%.3f %%, ".format(acc_list(0)(opt_layer_id_train) * 100) +
          "accuracy_test=%.3f %%".format(acc_list(1)(opt_layer_id_test) * 100))
      }

      reachMaxLayer = (layer_id == maxIteration) || outOfRounds
      if (reachMaxLayer) {
        timer.start("cal metric")
        val train_result = scanFeatureTrain.join(
          // join (predict feature col to predictionCol)
          lastPredictionTrain.withColumnRenamed(strategy.featuresCol, strategy.predictionCol),
          Seq(strategy.instanceCol, "rsp") // join on instanceCol
          // add a featureCol with featureCol + predictionCol
        ).drop(strategy.featuresCol)
          .withColumn(strategy.featuresCol, col(strategy.predictionCol)
          ).select(strategy.instanceCol, strategy.featuresCol, strategy.labelCol).toDF().cache()
        val test_result = scanFeatureTest.join(
          // join (predict feature col to predictionCol)
          lastPredictionTest.withColumnRenamed(strategy.featuresCol, strategy.predictionCol),
          Seq(strategy.instanceCol, "rsp") // join on instanceCol
          // add a featureCol with featureCol + predictionCol
        ).drop(strategy.featuresCol)
          .withColumn(strategy.featuresCol, col(strategy.predictionCol)
          ).select(strategy.instanceCol, strategy.featuresCol, strategy.labelCol).toDF().cache()

        //        train_result.count
        //        test_result.count
        logInfo("reach max layer, train_result and test_result calculating......")


        val train_acc = gcForestEvaluator.evaluateAccuracyPartition(train_result)
        val test_acc = gcForestEvaluator.evaluateAccuracyPartition(test_result)
        val train_precision = gcForestEvaluator.evaluatePrecisionPartition(train_result, strategy.classNum)
        val test_precision = gcForestEvaluator.evaluatePrecisionPartition(test_result, strategy.classNum)
        val train_f1score = gcForestEvaluator.evaluateF1ScorePartition(train_result, strategy.classNum)
        val test_f1score = gcForestEvaluator.evaluateF1ScorePartition(test_result, strategy.classNum)
        val train_kappa = gcForestEvaluator.evaluateKappaPartition(train_result, strategy.classNum)
        val test_kappa = gcForestEvaluator.evaluateKappaPartition(test_result, strategy.classNum)

        logInfo(s"[$getNowTime] rsp layer [${layer_id}] gcForests summary" +
          s"[Result]  =  " +
          "accuracy_train=%.3f %%, ".format(train_acc.getAccuracy * 100) +
          "accuracy_test=%.3f %%".format(test_acc.getAccuracy * 100) + "  " +
          "precision_train=%.3f %%, ".format(train_precision.getPrecision * 100) +
          "precision_test=%.3f %%".format(test_precision.getPrecision * 100) + "  " +
          "F1-Score_train=%.3f %%, ".format(train_f1score.getF1Score * 100) +
          "F1-Score_test=%.3f %%".format(test_f1score.getF1Score * 100) + "  " +
          "Kappa_train=%.3f %%, ".format(train_kappa.getKappa * 100) +
          "Kappa_test=%.3f %%".format(test_kappa.getKappa * 100)
        )
        timer.stop("cal metric")

        train_result.unpersist()
        test_result.unpersist()

      }
      logInfo(s"[$getNowTime] rsp-gcforest Layer $layer_id" +
        s" time cost: ${(System.currentTimeMillis() - stime) / 1000.0} s")

      training.unpersist()
      testing.unpersist()
      transformedResults.foreach { t =>
        t._1.unpersist()
        t._2.unpersist()
      }
      for (i <- 0 to (strategy.rfNum + strategy.crfNum) - 1) {
        for (j <- 0 to r - 1) {
          sampleTrain(i)(j).unpersist()
        }
      }
//      sc.getPersistentRDDs.foreach(turple => {
//        val xx = turple._2.toString()
//        logInfo("xx " + xx)
//      })
//      logInfo(s"sc.getPersistentRDDs.size:${sc.getPersistentRDDs.size}")
      layer_id += 1
    }
    // end to train the layer ${layer_id}, loop



    logger.info("shutdown executors")
    executors.shutdown()
    logger.info(s"[$getNowTime] Cascade Forest Training Finished!")

    scanFeatureTrain.unpersist()
    scanFeatureTest.unpersist()
    lastPredictionTrain.unpersist()
    lastPredictionTest.unpersist()

    timer.stop("total")

    logger.info(s"[$getNowTime] Internal timing for RSP-GCForestImpl:")
    logger.info(s"$timer")
//    sc.getPersistentRDDs.foreach(turple => {
//      val xx = turple._2.toString()
//      logInfo("last xx" + xx)
//    })
    logInfo(s"PersistentRDDs.size:${sc.getPersistentRDDs.size}")
    // scalastyle:on println
    new RSPGCForestClassificationModel(erfModels.toArray, numClasses)
  }
}


class RSPGCForestTask4(spark: SparkSession, gcForest: GCForestClassifier, training: Dataset[Row], testing: Dataset[Row], sampleTrain: Array[Array[Dataset[Row]]], rfIndex: Int)
  extends Callable[(Dataset[Row], Dataset[Row], GCForestClassificationModel)] with Logging {
  override def call(): (Dataset[Row], Dataset[Row], GCForestClassificationModel) = {
    val sc = spark.sparkContext
    val parallelism = Engine.getParallelism(spark.sparkContext)
    //    logInfo(s"Total Cores is $parallelism")
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.sparkContext.getConf.registerKryoClasses(Array(classOf[RandomForestClassifier]))

    val model = gcForest.train(sampleTrain(rfIndex))


    val transformedTrain = model.transform(training).coalesce(sc.defaultParallelism).cache()
    val transformedTest = model.transform(testing).coalesce(sc.defaultParallelism).cache()
    transformedTrain.count
    transformedTest.count
    logDebug(s"multithread training gcforest Finished!")
    (transformedTrain, transformedTest, model)
  }
}


