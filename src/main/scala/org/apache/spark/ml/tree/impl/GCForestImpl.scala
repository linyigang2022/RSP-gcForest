/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.tree.impl

import org.apache.log4j.Logger
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{Accuracy, Metric, gcForestEvaluator}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT}
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

import java.text.SimpleDateFormat
import java.util.Date
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
 * RSP used GCForestImpl
 */
private[spark] object GCForestImpl extends Logging {

  def run(
           input: Dataset[_],
           gcforestStategy: GCForestStrategy
         ): GCForestClassificationModel = {
    train(input, strategy = gcforestStategy)
  }

//  def runWithValidation(
//                         input: Dataset[_],
//                         validationInput: Dataset[_],
//                         gCForestStrategy: GCForestStrategy
//                       ): GCForestClassificationModel = {
//    trainWithValidation(input, validationInput, gCForestStrategy)
//  }


  def runWithValidation(
                         rsp: Array[DataFrame],
                         gCForestStrategy: GCForestStrategy
                       ): GCForestClassificationModel = {
    trainWithValidation(rsp, gCForestStrategy)
  }
  val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS")


  /**
   * train a sub random forest of gcforest
   *
   * @param training train set
   * @param rfc      the random forest classifier to be trained
   * @param numFolds
   * @param seed     random seed
   * @param strategy gcforeststrategy
   * @param isScan
   * @param message
   * @return
   */
  def cvClassVectorGenerator(
                              training: Dataset[_],
                              rfc: RandomForestClassifier,
                              numFolds: Int,
                              seed: Long,
                              strategy: GCForestStrategy,
                              isScan: Boolean = false,
                              message: String = ""):
  (DataFrame, Metric, RandomForestClassificationModel) = {
    val schema = training.schema
    val sparkSession = training.sparkSession
    var out_train: DataFrame = null // closure need

    // cross-validation for k classes distribution feature
    var train_metric = new Accuracy(0, 0)
    val splits = MLUtils.kFold(training.toDF().rdd, numFolds, seed * System.currentTimeMillis())
    splits.zipWithIndex.foreach {
      case ((t, v), splitIndex) =>
        val trainingDataset = sparkSession.createDataFrame(t, schema)
        val validationDataset = sparkSession.createDataFrame(v, schema)
        val model = rfc.fit(trainingDataset)

        trainingDataset.unpersist()
        // rawPrediction == probabilityCol
        val val_result = model.transform(validationDataset)
          .drop(strategy.featuresCol)
          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        out_train = if (out_train == null) val_result else out_train.union(val_result)
        if (!isScan) {
          val val_acc = gcForestEvaluator.evaluateAccuracyPartition(val_result)
          train_metric += val_acc
          // scalastyle:off println
          //          logInfo(s"[$getNowTime] $message ${numFolds}_folds.train_$splitIndex = $val_acc")
          // scalastyle:on println
        }
        validationDataset.unpersist()
    }
    (out_train, train_metric, rfc.fit(training))
  }

  // create a random forest classifier by type
  def genRFClassifier(rfType: String,
                      strategy: GCForestStrategy,
                      isScan: Boolean,
                      num: Int): RandomForestClassifier = {
    val rf = rfType match {
      case "rfc" => new RandomForestClassifier().setFeatureSubsetStrategy(strategy.featureSubsetStrategy)
      case "crfc" => new CompletelyRandomForestClassifier().setFeatureSubsetStrategy(strategy.crf_featureSubsetStrategy)
    }

    rf.setNumTrees(if (isScan) strategy.scanForestTreeNum else strategy.cascadeForestTreeNum)
      .setMaxBins(strategy.maxBins)
      .setMaxDepth(strategy.maxDepth)
      .setMinInstancesPerNode(if (isScan) strategy.scanMinInsPerNode
      else strategy.cascadeMinInsPerNode)
      .setMinInfoGain(strategy.minInfoGain)
      // .setFeatureSubsetStrategy(strategy.featureSubsetStrategy)
      .setCacheNodeIds(strategy.cacheNodeId)
      .setMaxMemoryInMB(strategy.maxMemoryInMB)
      .setSeed(System.currentTimeMillis() + num * 123L + rfType.hashCode % num)
  }


  /**
   *
   * @param training  train set
   * @param testing   test set
   * @param rfc_class the type of random forest, random forest or completely random forest
   * @param numFolds  numfolds of cross-validation
   * @param seed      random seed
   * @param timer
   * @param strategy  gcforeststrategy, including some training strategy
   * @param isScan
   * @param layer_id
   * @param estimator_id
   * @return
   */
  def cvClassVectorGeneratorWithValidation(
                                            training: Dataset[_],
                                            rfc_class: String,
                                            numFolds: Int,
                                            seed: Long,
                                            timer: TimeTracker,
                                            strategy: GCForestStrategy,
                                            isScan: Boolean,
                                            layer_id: Int,
                                            estimator_id: Int):
  (DataFrame, Metric, RandomForestClassificationModel) = {
    val logger = Logger.getLogger(this.getClass)
    val schema = training.schema
//    require(schema.equals(testing.schema))
    val message = s"layer [$layer_id] - estimator [$estimator_id]"

    // cross-validation for k classes distribution features, in blbGCForest, we cancel the cross-validation of subrandomforest of gcforest
    var train_metric = new Accuracy(0, 0)
    var test_metric = new Accuracy(0, 0)
    timer.start("Kfold cancel, fit sub random forest")
    val rfc = genRFClassifier(rfc_class, strategy, isScan = isScan, num = 1)
//    training.cache()
//    testing.cache()
//    synchronized{
//      println(s"training:(${training.count},${training.first().asInstanceOf[Row].mkString.split(",").length})")
//      println(s"testing:(${testing.count},${testing.first().asInstanceOf[Row].mkString.split(",").length})")
//    }
//    training.sparkSession.sparkContext.getPersistentRDDs.foreach(truple => {
//      val xx = truple._2.toString()
//      println(xx)
//    })
//    println("---------------------------------------------------------")
    logDebug("Sub Random Forest starts to train")
    val model = rfc.fit(training)
    logDebug("The fitting of the sub random forest has finished")

    // calculate the accuracy
    // rawPrediction == probabilityCol
    val out_train = model.transform(training)
      .drop(strategy.featuresCol)
      .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
    train_metric += gcForestEvaluator.evaluateAccuracyPartition(out_train)
    logDebug(s"[$getNowTime] randomforest of gcforest, training accuracy：$train_metric")

//    val out_test = model.transform(testing)
//      .drop(strategy.featuresCol)
//      .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
//    test_metric += gcForestEvaluator.evaluateAccuracyPartition(out_test)
//    logDebug(s"[$getNowTime] randomforest of gcforest, testing accuracy：$test_metric")

    timer.stop("Kfold cancel, fit sub random forest")
    (out_train, train_metric, model)
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
        Seq(strategy.instanceCol) // join on instanceCol
        // add a featureCol with featureCol + predictionCol
      ).withColumn(strategy.featuresCol, vectorMerge(col(strategy.featuresCol),
        col(strategy.predictionCol))
      ).select(strategy.instanceCol, strategy.featuresCol, strategy.labelCol).toDF()
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
   * train a GCForestClassificationModel without validation
   *
   * @param input
   * @param strategy
   * @return
   */
  @deprecated
  def train(
             input: Dataset[_],
             strategy: GCForestStrategy): GCForestClassificationModel = {
    val numClasses: Int = strategy.classNum
    val erfModels = ArrayBuffer[Array[RandomForestClassificationModel]]()
    val n_train = input.count()

    val scanFeature_train = multi_grain_Scan(input, strategy)

    scanFeature_train.cache()
    // scalastyle:off println
    logInfo(s"[$getNowTime] Cascade Forest begin!")

    val sparkSession = scanFeature_train.sparkSession
    val sc = sparkSession.sparkContext
    val rng = new Random()
    rng.setSeed(System.currentTimeMillis())

    var lastPrediction: DataFrame = null
    val acc_list = ArrayBuffer[Double]()

    // Init classifiers
    val maxIteration = strategy.maxIteration
    require(maxIteration > 0, "Non-positive maxIteration")
    var layer_id = 1
    var reachMaxLayer = false

    while (!reachMaxLayer) {

      logInfo(s"[$getNowTime] Training Cascade Forest Layer $layer_id")

      val randomForests = (
        Range(0, 4).map(it =>
          genRFClassifier("rfc", strategy, isScan = false, num = rng.nextInt + it))
          ++
          Range(4, 8).map(it =>
            genRFClassifier("crfc", strategy, isScan = false, num = rng.nextInt + it))
        ).toArray[RandomForestClassifier]
      assert(randomForests.length == 8, "random Forests inValid!")
      // scanFeatures_*: (instanceId, label, features)
      val training = mergeFeatureAndPredict(scanFeature_train, lastPrediction, strategy)
        .persist(StorageLevel.MEMORY_ONLY_SER)
      val bcastTraining = sc.broadcast(training)
      val features_dim = training.first().mkString.split(",").length

      logInfo(s"[$getNowTime] Training Set = ($n_train, $features_dim)")

      var ensemblePredict: DataFrame = null // closure need

      var layer_train_metric: Accuracy = new Accuracy(0, 0) // closure need

      logInfo(s"[$getNowTime] Forests fitting and transforming ......")

      erfModels += randomForests.zipWithIndex.map { case (rf, it) =>
        val transformed = cvClassVectorGenerator(
          bcastTraining.value, rf, strategy.numFolds, strategy.seed, strategy,
          isScan = false, s"layer [$layer_id] - estimator [$it]")
        val predict = transformed._1
          .withColumn(strategy.forestIdCol, lit(it))
          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)

        ensemblePredict =
          if (ensemblePredict == null) predict else ensemblePredict.union(predict)

        layer_train_metric = layer_train_metric + transformed._2

        logInfo(s"[$getNowTime] [Estimator Summary] " +
          s"layer [$layer_id] - estimator [$it] Train.predict = ${transformed._2}")
        transformed._3
      }

      logInfo(s"[$getNowTime] [Layer Summary] layer [$layer_id] - " +
        s"train.classifier.average = ${layer_train_metric.div(8d)}")
      logInfo(s"[$getNowTime] Forests fitting and transforming finished!")

      val schema = new StructType()
        .add(StructField(strategy.instanceCol, LongType))
        .add(StructField(strategy.featuresCol, new VectorUDT))

      logInfo(s"[$getNowTime] Getting prediction RDD ......")

      acc_list += layer_train_metric.getAccuracy

      val predictRDDs = {
        val grouped = ensemblePredict.rdd.groupBy(_.getAs[Long](strategy.instanceCol))
        grouped.map { group =>
          val instanceId = group._1
          val rows = group._2
          val features = new DenseVector(rows.toArray
            .sortWith(_.getAs[Int](strategy.forestIdCol) < _.getAs[Int](strategy.forestIdCol))
            .flatMap(_.getAs[Vector](strategy.featuresCol).toArray))
          Row.fromSeq(Array[Any](instanceId, features))
        }
      }

      // predictRDDs.foreach(r => r.persist(StorageLevel.MEMORY_ONLY_SER))
      logInfo(s"[$getNowTime] Get prediction RDD finished! Layer $layer_id training finished!")

      val opt_layer_id_train = acc_list.zipWithIndex.maxBy(_._1)._2


      if (opt_layer_id_train + 1 == layer_id) {
        logInfo(s"[$getNowTime] [Result] [Optimal Layer] max_layer_num = $layer_id " +
          s"accuracy_train = ${acc_list(opt_layer_id_train) * 100}%")
      }
      lastPrediction = sparkSession.createDataFrame(predictRDDs, schema)
      val outOfRounds = layer_id - opt_layer_id_train >= strategy.earlyStoppingRounds
      if (outOfRounds) {
        logInfo(s"[$getNowTime] " +
          s"outOfRounds [Result][Optimal Level Detected] opt_layer_id = " +
          s"$opt_layer_id_train, " +
          s"accuracy_train=${acc_list(opt_layer_id_train)}")
      }
      reachMaxLayer = (layer_id == maxIteration) || outOfRounds
      if (reachMaxLayer) {
        logInfo(s"[$getNowTime] " +
          s"[Result][Reach Max Layer] max_layer_num=$layer_id, " +
          s"accuracy_train=$layer_train_metric")
      }
      layer_id += 1
    }

    scanFeature_train.unpersist()

    //      logInfo(s"[$getNowTime] Cascade Forest Training Finished!")
    // scalastyle:on println
    new GCForestClassificationModel(erfModels.toArray, numClasses)
  }

  /**
   * Train a Cascade Forest with validation
   */
  private def trainWithValidation(
                                   rsp: Array[DataFrame],
                                   strategy: GCForestStrategy): GCForestClassificationModel = {
    val logger = Logger.getLogger(this.getClass)
    // multi grain scan. For structured data, multi-granularity scanning will return directly.
    val timer = new TimeTracker()
    timer.start("total")
    if (strategy.idebug) logInfo(s"[$getNowTime] timer.start(total)")
    val numClasses: Int = strategy.classNum
    // TODO: better representation to model
    val erfModels = ArrayBuffer[Array[RandomForestClassificationModel]]() // layer - (forest * fold)
    val n_train = rsp(0)

    val scanFeature = Array.ofDim[DataFrame](strategy.rfNum + strategy.crfNum)
//    println(s"gcforest strategy.rfNum + strategy.crfNum:${strategy.rfNum + strategy.crfNum}")
    for (j <- 0 until strategy.rfNum + strategy.crfNum) {
      scanFeature(j) = multi_grain_Scan(rsp(j), strategy)
    }



    //    scanFeature_train.cache()
    //    scanFeature_test.cache()

    //    logger.info(s"[$getNowTime] Cascade Forest begin!")

    // Init
    timer.start("init")
    val sparkSession = scanFeature(0).sparkSession
    val sc = sparkSession.sparkContext
    val rng = new Random()
    rng.setSeed(System.currentTimeMillis())



    var lastPrediction: DataFrame = null // closure need
    var lastPrediction_test: DataFrame = null // closure need

    val acc_list = Array(ArrayBuffer[Double](), ArrayBuffer[Double]())
    var ensemblePredict: DataFrame = null // closure need
    var ensemblePredict_test: DataFrame = null // closure need

    var layer_train_metric: Accuracy = new Accuracy(0, 0) // closure need
    var layer_test_metric: Accuracy = new Accuracy(0, 0) // closure need

    val maxIteration = strategy.maxIteration
    require(maxIteration > 0, "Non-positive maxIteration")
    var layer_id = 1
    var reachMaxLayer = false

    timer.stop("init")

    while (!reachMaxLayer) {

      val stime = System.currentTimeMillis()

      //      logger.info(s"[$getNowTime] Training Cascade Forest Layer $layer_id")

      // create random forest classifier based on random forest type
      val randomForests = (
        Range(0, strategy.rfNum).map(_ => "rfc")
          ++
          Range(strategy.rfNum, strategy.rfNum + strategy.crfNum).map(_ => "crfc")
        ).toArray[String]
      assert(randomForests.length == strategy.rfNum + strategy.crfNum, "random Forests inValid!")
      // scanFeatures_*: (instanceId, label, features)

      timer.start("merge to produce training, testing and persist")


      var mergedData = Array.ofDim[Dataset[Row]](strategy.rfNum + strategy.crfNum)
      for (j <- 0 until strategy.rfNum + strategy.crfNum) {
        mergedData(j) = mergeFeatureAndPredict(scanFeature(j), lastPrediction, strategy).coalesce(sc.defaultParallelism)
      }

      timer.stop("merge to produce training, testing and persist")

      val features_dim = mergedData(0).first().mkString.split(",").length // action, get training truly

//      logger.info(s"gcforest: Training Set = ($n_train, $features_dim), " +
//        s"Testing Set = ($n_test, $features_dim)")


      ensemblePredict = null

      layer_train_metric.reset()
      layer_test_metric.reset()

//      val n = training.count().toDouble


      //      logger.info(s"before gcforest sub random forest sampling, the size of train set: ${training.count()},after sampling: ${sampleTraining(0).count()}")
      //      logger.debug(s"sub random forest of gcforest：${strategy.rfNum + strategy.crfNum}")
      //      sampleTraining.foreach { training =>
      //        training.cache()
      //        training.first().length
      //      }
      //      sampleTesting.foreach { testing =>
      //        testing.cache()
      //        testing.first().length
      //      }

      //      logInfo(s"[$getNowTime] Forests fitting and transforming ......")
      timer.start("randomForests training")
      if (strategy.idebug) logInfo(s"[$getNowTime] timer.start(randomForests training)")

      // train the randomforests in layer [layer_id]
      erfModels += randomForests.zipWithIndex.map { case (rf_type, it) =>

        val transformed = cvClassVectorGeneratorWithValidation(
          mergedData(it), rf_type, strategy.numFolds, strategy.seed, timer, strategy,
          isScan = false, layer_id, it)

        // calculate the training accuracy and testing accuracy
        val predict = transformed._1
          .withColumn(strategy.forestIdCol, lit(it))
          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)


        ensemblePredict =
          if (ensemblePredict == null) predict else ensemblePredict.union(predict)


        if (strategy.idebug) logInfo(s"[$getNowTime] timer.stop(add forestIdCol and Union)")

        layer_train_metric = layer_train_metric + transformed._2
//        layer_test_metric = layer_test_metric + transformed._4

        logDebug(s"[$getNowTime] [Estimator Summary] " +
          s"layer [$layer_id] - estimator [$it] Train.predict = ${transformed._3}")
//        logDebug(s"[$getNowTime] [Estimator Summary] " +
//          s"layer [$layer_id] - estimator [$it]  Test.predict = ${transformed._4}")

        transformed._3
      }
      timer.stop("randomForests training")

      acc_list(0) += layer_train_metric.getAccuracy
      acc_list(1) += layer_test_metric.getAccuracy

      val schema = new StructType()
        .add(StructField(strategy.instanceCol, LongType))
        .add(StructField(strategy.featuresCol, new VectorUDT))

      // get predictRDD, [instanceId, [features1, ..., features${rfNum}]]      //feature = probability
      logger.debug(s"[$getNowTime] Getting prediction RDD ......")
      timer.start("flatten prediction")
      val predict = ensemblePredict
      val predictRDDs = {
          val grouped = predict.rdd.groupBy(_.getAs[Long](strategy.instanceCol))
          val predictRDD = grouped.map { group =>
            val instanceId = group._1
            val rows = group._2
            val features = new DenseVector(rows.toArray
              .sortWith(_.getAs[Int](strategy.forestIdCol) < _.getAs[Int](strategy.forestIdCol))
              .flatMap(_.getAs[Vector](strategy.featuresCol).toArray))
            Row.fromSeq(Array[Any](instanceId, features))
          }
          predictRDD
        }


      timer.stop("flatten prediction")

      logDebug(s"[$getNowTime] Get prediction RDD finished! Layer $layer_id training finished!")

      val opt_layer_id_train = acc_list(0).zipWithIndex.maxBy(_._1)._2

      // early stop, if there is no improvement in accuracy for "earlyStoppingRounds".

      reachMaxLayer = (layer_id == maxIteration)
      if (reachMaxLayer) {
        logInfo(s"[Result][gcForest]" +
          s"accuracy_train=$layer_train_metric, accuracy_test=$layer_test_metric")
      }
      logDebug(s"[$getNowTime] Layer $layer_id" +
        s" time cost: ${(System.currentTimeMillis() - stime) / 1000.0} s")
      layer_id += 1
    }
    //    scanFeature_train.unpersist()
    //    scanFeature_test.unpersist()

    logDebug(s"[$getNowTime] Cascade Forest Training Finished!")
    timer.stop("total")
    if (strategy.idebug) logInfo(s"[$getNowTime] timer.stop(total)")

    //    logInfo(s"[$getNowTime] Internal timing for GCForestImpl:")
    new GCForestClassificationModel(erfModels.toArray, numClasses)
  }
}
