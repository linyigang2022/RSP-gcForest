package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.tree.impl.{GCForestImpl}

class GCForestClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GCForestClassifier, GCForestClassificationModel]
    with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcf"))

  override def setNumClasses(value: Int): this.type = set(classNum, value)

  override def setModelPath(value: String): GCForestClassifier.this.type = set(modelPath, value)

  override def setDataSize(value: Array[Int]): this.type = set(dataSize, value)

  override def setDataStyle(value: String): this.type = set(dataStyle, value)

  override def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  override def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  override def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  override def setMaxIteration(value: Int): this.type = set(MaxIteration, value)

  override def setEarlyStoppingRounds(value: Int): this.type = set(earlyStoppingRounds, value)

  override def setIDebug(value: Boolean): GCForestClassifier.this.type = set(idebug, value)

  override def setSubRFNum(value: Int): GCForestClassifier.this.type = set(subRFNum, value)

  override def setMaxDepth(value: Int): this.type = set(MaxDepth, value)

  override def setMaxBins(value: Int): GCForestClassifier.this.type = set(MaxBins, value)

  override def setMinInfoGain(value: Double): GCForestClassifier.this.type = set(minInfoGain, value)

  override def setScanForestMinInstancesPerNode(value: Int):
  GCForestClassifier.this.type = set(scanMinInsPerNode, value)

  override def setCascadeForestMinInstancesPerNode(value: Int):
  GCForestClassifier.this.type = set(cascadeMinInsPerNode, value)

  override def setFeatureSubsetStrategy(value: String):
  GCForestClassifier.this.type = set(featureSubsetStrategy, value)

  override def setCacheNodeId(value: Boolean):
  GCForestClassifier.this.type = set(cacheNodeId, value)

  override def setMaxMemoryInMB(value: Int):
  GCForestClassifier.this.type = set(maxMemoryInMB, value)

  override def setRFNum(value: Int): GCForestClassifier.this.type = set(rfNum, value)

  override def setCRFNum(value: Int): GCForestClassifier.this.type = set(crfNum, value)

  def getGCForestStrategy: GCForestStrategy = {
    GCForestStrategy($(classNum), $(modelPath), $(multiScanWindow),
      $(dataSize), $(rfNum), $(crfNum),
      $(scanForestTreeNum), $(cascadeForestTreeNum), $(scanMinInsPerNode),
      $(cascadeMinInsPerNode), $(featureSubsetStrategy), $(crf_featureSubsetStrategy), $(MaxBins),
      $(MaxDepth), $(minInfoGain), $(MaxIteration), $(maxMemoryInMB),
      $(numFolds), $(earlyStoppingRounds),
      $(earlyStopByTest), $(dataStyle), $(seed), $(cacheNodeId),
      $(windowCol), $(scanCol), $(forestIdCol), $(idebug),
      $(subRFNum))
  }

  def getDefaultStrategy: GCForestStrategy = {
    GCForestStrategy(2, $(modelPath), Array(), Array(113), idebug = false)
  }



  def train(rspRF: Array[Dataset[Row]]): GCForestClassificationModel = {
    val casted_rsp = rspRF.map { rspSRF => {
      val trainset = rspSRF
      transformSchema(trainset.schema, logging = true)

      // Cast LabelCol to DoubleType and keep the metadata.
      val labelMeta = trainset.schema($(labelCol)).metadata
      val casted_train =
        trainset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
      casted_train
    }

    }
    // This handles a few items such as schema validation.
    // Developers only need to implement train().


    copyValues(GCForestImpl.runWithValidation(casted_rsp, getGCForestStrategy))
  }

  override def train(dataset: Dataset[_]): GCForestClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(dataset.schema, logging = true)

    // Cast LabelCol to DoubleType and keep the metadata.
    val labelMeta = dataset.schema($(labelCol)).metadata
    val casted_train =
      dataset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
    GCForestImpl.run(casted_train, getGCForestStrategy)
  }

  override def copy(extra: ParamMap): GCForestClassifier = defaultCopy(extra)
}


private[ml] class GCForestClassificationModel(
                                               override val uid: String,
                                               private val cascadeForest: Array[Array[RandomForestClassificationModel]],
                                               override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GCForestClassificationModel]
    with GCForestParams with MLWritable with Serializable {

  def this(
            cascadeForest: Array[Array[RandomForestClassificationModel]],
            numClasses: Int) =
    this(Identifiable.randomUID("gcfc"), cascadeForest, numClasses)

  val numCascades: Int = cascadeForest.length


  //  override def predictRaw(features: Vector): Vector = features
  override def predictRaw(features: Vector): Vector = {
    var scanFeatures: Vector = null
    if ($(dataStyle) == "Seq") {
      scanFeatures = features
    }
    val avgPredict = Array.fill[Double](numClasses)(0d)
    var lastPredict = Array[Double]()

    cascadeForest.foreach { models =>
      lastPredict = models.flatMap(
        m => m.predictProbability(new DenseVector(features.toArray.union(lastPredict))).toArray
      )
    }

    lastPredict.indices.foreach { i =>
      val classType = i % numClasses
      avgPredict(classType) = avgPredict(classType) + lastPredict(i)
    }

    new DenseVector(avgPredict)
  }

  override def predictProbability(features: Vector): Vector = super.predictProbability(features)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case _: SparseVector =>
        throw new RuntimeException("Unexpected error in GCForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def copy(extra: ParamMap): GCForestClassificationModel = {
    copyValues(new GCForestClassificationModel(uid, cascadeForest, numClasses), extra)
  }

  override def write: MLWriter =
    new GCForestClassificationModel.GCForestClassificationModelWriter(this)
}


object GCForestClassificationModel extends MLReadable[GCForestClassificationModel] {
  override def read: MLReader[GCForestClassificationModel] = new GCForestClassificationModelReader

  override def load(path: String): GCForestClassificationModel = super.load(path)

  private[GCForestClassificationModel]
  class GCForestClassificationModelWriter(instance: GCForestClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {

      val gcMetadata: JObject = Map(
        "numClasses" -> instance.numClasses,
        "numCascades" -> instance.numCascades)
      DefaultParamsWriter.saveMetadata(instance, path, sparkSession.sparkContext, Some(gcMetadata))

      // CascadeForestModel
      val cascadePath = new Path(path, "cascade").toString
      instance.cascadeForest.zipWithIndex.foreach { case (models, level) =>
        val modelsPath = new Path(cascadePath, level.toString).toString
        models.zipWithIndex.foreach { case (model, index) =>
          val modelPath = new Path(modelsPath, index.toString).toString
          model.save(modelPath)
        }
      }
    }
  }

  private class GCForestClassificationModelReader
    extends MLReader[GCForestClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GCForestClassificationModel].getName

    override def load(path: String): GCForestClassificationModel = {
      implicit val format = DefaultFormats
      val gcMetadata = DefaultParamsReader.loadMetadata(path, sparkSession.sparkContext, className)

      val numClasses = (gcMetadata.metadata \ "numClasses").extract[Int]
      val numScans = (gcMetadata.metadata \ "numScans").extract[Int]
      val numCascades = (gcMetadata.metadata \ "numCascades").extract[Int]

      val scanPath = new Path(path, "scan").toString

      val cascadePath = new Path(path, "cascade").toString
      val cascadeForest = Range(0, numCascades).map { level =>
        val modelsPath = new Path(cascadePath, level.toString).toString
        Range(0, 4).map { index =>
          val modelPath = new Path(modelsPath, index.toString).toString
          RandomForestClassificationModel.load(modelPath)
        }.toArray
      }.toArray

      val gcForestModel =
        new GCForestClassificationModel(gcMetadata.uid, cascadeForest, numClasses)

      //      DefaultParamsReader.getAndSetParams(gcForestModel, gcMetadata)
      gcMetadata.getAndSetParams(gcForestModel)
      gcForestModel
    }
  }
}

