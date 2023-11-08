package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.ml.tree.impl.RSPGCForestImpl
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.json4s.DefaultFormats

/**
 * RSPGCForestClassifier
 *
 * @param uid
 */
class RSPGCForestClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, RSPGCForestClassifier, RSPGCForestClassificationModel]
    with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcf"))

  // method for setting RSPGCForestClassifier params
  override def setNumClasses(value: Int): this.type = set(classNum, value)

  override def setModelPath(value: String): RSPGCForestClassifier.this.type = set(modelPath, value)

  override def setDataSize(value: Array[Int]): this.type = set(dataSize, value)

  override def setDataStyle(value: String): this.type = set(dataStyle, value)

  override def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  override def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  override def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  override def setMaxIteration(value: Int): this.type = set(MaxIteration, value)

  override def setEarlyStoppingRounds(value: Int): this.type = set(earlyStoppingRounds, value)

  override def setIDebug(value: Boolean): RSPGCForestClassifier.this.type = set(idebug, value)

  override def setSubRFNum(value: Int): RSPGCForestClassifier.this.type = set(subRFNum, value)

  override def setLambda(value: Double): RSPGCForestClassifier.this.type = set(lambda, value)

  override def setNewFeatures(value: String): RSPGCForestClassifier.this.type = set(newFeatures, value)

  override def setDataset(value: String): RSPGCForestClassifier.this.type = set(dataset, value)

  override def setMaxDepth(value: Int): this.type = set(MaxDepth, value)

  override def setMaxBins(value: Int): RSPGCForestClassifier.this.type = set(MaxBins, value)

  override def setMinInfoGain(value: Double): RSPGCForestClassifier.this.type = set(minInfoGain, value)

  override def setScanForestMinInstancesPerNode(value: Int):
  RSPGCForestClassifier.this.type = set(scanMinInsPerNode, value)

  override def setCascadeForestMinInstancesPerNode(value: Int):
  RSPGCForestClassifier.this.type = set(cascadeMinInsPerNode, value)

  override def setFeatureSubsetStrategy(value: String):
  RSPGCForestClassifier.this.type = set(featureSubsetStrategy, value)

  override def setCacheNodeId(value: Boolean):
  RSPGCForestClassifier.this.type = set(cacheNodeId, value)

  override def setMaxMemoryInMB(value: Int):
  RSPGCForestClassifier.this.type = set(maxMemoryInMB, value)

  override def setRFNum(value: Int): RSPGCForestClassifier.this.type = set(rfNum, value)

  override def setCRFNum(value: Int): RSPGCForestClassifier.this.type = set(crfNum, value)

  /**
   * get Strategy of the RSPGCForestClassifier
   */
  def getGCForestStrategy: GCForestStrategy = {
    GCForestStrategy($(classNum), $(modelPath), $(multiScanWindow),
      $(dataSize), $(rfNum), $(crfNum),
      $(scanForestTreeNum), $(cascadeForestTreeNum), $(scanMinInsPerNode),
      $(cascadeMinInsPerNode), $(featureSubsetStrategy), $(crf_featureSubsetStrategy), $(MaxBins),
      $(MaxDepth), $(minInfoGain), $(MaxIteration), $(maxMemoryInMB),
      $(numFolds), $(earlyStoppingRounds),
      $(earlyStopByTest), $(dataStyle), $(seed), $(cacheNodeId),
      $(windowCol), $(scanCol), $(forestIdCol), $(idebug),
      $(subRFNum), $(lambda), $(newFeatures), $(dataset))
  }

  def getDefaultStrategy: GCForestStrategy = {
    GCForestStrategy(2, $(modelPath), Array(), Array(113), idebug = false)
  }

  // train the RSPGCForestClassifier with validation and convert it to a RSPGCForestClassificationModel
  //  def train(trainset: Dataset[_], testset: Dataset[_]): RSPGCForestClassificationModel = {
  //    // This handles a few items such as schema validation.
  //    // Developers only need to implement train().
  //    transformSchema(trainset.schema, logging = true)
  //    transformSchema(testset.schema, logging = true)
  //
  //    // Cast LabelCol to DoubleType and keep the metadata.
  //    val labelMeta = trainset.schema($(labelCol)).metadata
  //    val casted_train = trainset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
  //    val labelMeta_test = testset.schema($(labelCol)).metadata
  //    val casted_test = testset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta_test)
  //
  //    // RSPFCForestImpl is a object that implement training details
  //    copyValues(RSPGCForestImpl.runWithValidation(casted_train, casted_test, getGCForestStrategy))
  //  }

  def train(selectedRSP: Array[Dataset[_]], fullTest: Dataset[_]): RSPGCForestClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    // RSPFCForestImpl is a object that implement training details
    val castedRSP = selectedRSP.map {
      rspRF =>
        val trainset = rspRF
        transformSchema(trainset.schema, logging = true)
        val labelMeta = trainset.schema($(labelCol)).metadata
        val casted_train = trainset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
        casted_train
    }
    val labelMeta_test = fullTest.schema($(labelCol)).metadata
    val casted_test = fullTest.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta_test)
    copyValues(RSPGCForestImpl.runWithValidation(castedRSP, casted_test, getGCForestStrategy))
  }


  // train the RSPGCForestClassifier without validation and convert it to a RSPGCForestClassificationModel
  override def train(dataset: Dataset[_]): RSPGCForestClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(dataset.schema, logging = true)

    // Cast LabelCol to DoubleType and keep the metadata.
    val labelMeta = dataset.schema($(labelCol)).metadata
    val casted_train =
      dataset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
    RSPGCForestImpl.run(casted_train, getGCForestStrategy)
  }

  override def copy(extra: ParamMap): RSPGCForestClassifier = defaultCopy(extra)
}

// RSPGCForestClassificationModel is a trained RSPGCForestClassifier, fit data to classifier and it turn to a model
private[ml] class RSPGCForestClassificationModel(
                                                  override val uid: String,
                                                  private val gcForests: Array[Array[GCForestClassificationModel]],
                                                  override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, RSPGCForestClassificationModel]
    with GCForestParams with MLWritable with Serializable {

  def this(
            cascadeForest: Array[Array[GCForestClassificationModel]],
            numClasses: Int) =
    this(Identifiable.randomUID("gcfc"), cascadeForest, numClasses)

  val numGCForests: Int = gcForests.length

  /**
   * override def predictRaw(features: Vector): Vector = features
   *
   * @param features
   * @return probalility
   */
  override def predictRaw(features: Vector): Vector = {
    var scanFeatures: Vector = null
    if ($(dataStyle) == "Seq") {
      scanFeatures = features
    }
    val avgPredict = Array.fill[Double](numClasses)(0d)
    var lastPredict = Array[Double]()
    var stackedPredict = Array.fill[Double](numClasses)(0d)

    // n models will generate prediction results of length "n*classNum"
    gcForests.foreach { models =>
      lastPredict = models.flatMap(
        m => m.predictProbability(new DenseVector(features.toArray.union(lastPredict))).toArray
      )

      stackedPredict = Array.fill[Double](numClasses)(0d)
      lastPredict.indices.foreach { i =>
        val classType = i % numClasses
        stackedPredict(classType) = stackedPredict(classType) + lastPredict(i)
      }
      val n = lastPredict.length / numClasses

      lastPredict = stackedPredict.map(stackedPredict => {
        stackedPredict / n
      })
    }

    // stack the prediction results to length "classNum"
    lastPredict.indices.foreach { i =>
      val classType = i % numClasses
      avgPredict(classType) = avgPredict(classType) + lastPredict(i)
    }
//    println(lastPredict.mkString("="))
    new DenseVector(avgPredict)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case _: SparseVector =>
        throw new RuntimeException("Unexpected error in RSPGCForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  /**
   * transform a Dataset
   *
   * @param dataset
   * @return transformed DataFrame
   */
  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }


  override def write: MLWriter =
    new RSPGCForestClassificationModel.RSPGCForestClassificationModelWriter(this)


  override def copy(extra: ParamMap): RSPGCForestClassificationModel = {
    copyValues(new RSPGCForestClassificationModel(uid, gcForests, numClasses), extra)
  }
}


object RSPGCForestClassificationModel extends MLReadable[RSPGCForestClassificationModel] {
  override def read: MLReader[RSPGCForestClassificationModel] = new RSPGCForestClassificationModelReader

  override def load(path: String): RSPGCForestClassificationModel = super.load(path)

  private[RSPGCForestClassificationModel]
  class RSPGCForestClassificationModelWriter(instance: RSPGCForestClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      //todo implement the method to save the model
    }
  }

  private class RSPGCForestClassificationModelReader
    extends MLReader[RSPGCForestClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[RSPGCForestClassificationModel].getName

    override def load(path: String): RSPGCForestClassificationModel = {
      // todo: load model
      implicit val format = DefaultFormats
      val gcMetadata = DefaultParamsReader.loadMetadata(path, sparkSession.sparkContext, className)
      val numClasses = (gcMetadata.metadata \ "numClasses").extract[Int]
      new RSPGCForestClassificationModel(gcMetadata.uid, Array[Array[GCForestClassificationModel]](), numClasses)
    }
  }
}



