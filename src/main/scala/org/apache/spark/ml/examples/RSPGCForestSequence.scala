/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.examples

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem.getDefaultUri
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{RSPGCForestClassifier, RandomForestClassifier}
import org.apache.spark.ml.datasets._
import org.apache.spark.ml.util.engine.Engine
import org.apache.spark.sql.RspContext.NewRDDFunc
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.gcForestEvaluator
import org.apache.spark.ml.examples.Utils.TrainParams

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import java.io._
import java.util.Properties
import scala.util.Random

/**
 * RSPGCForestSequence is the entry class that applies RSP-GCForest to structured data
 */
object RSPGCForestSequence extends Logging {
  def main(args: Array[String]): Unit = {
    // setting parameters
    import Utils._
    // parse the paramaters
    val param = trainParser.parse(args, TrainParams()).get

    val stime = System.currentTimeMillis()

    val spark = if (param.isProvided) {
      SparkSession
        .builder()
        .appName(this.getClass.getSimpleName)
        .getOrCreate()
    } else {
      SparkSession
        .builder()
        .appName(this.getClass.getSimpleName)
        .master("local[*]")
        .getOrCreate()
    }

    // obtain parallelism
    val parallelism = Engine.getParallelism(spark.sparkContext)

    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.sparkContext.getConf.registerKryoClasses(Array(classOf[RandomForestClassifier]))




    // set the log level
    println(s"new features: ${param.newFeatures}")
    // set the log level
    val level = param.debugLevel match {
      case "INFO" => Level.INFO
      case "ERROR" => Level.ERROR
    }
    spark.sparkContext.setLogLevel("ERROR")
    Logger.getLogger("org.apache.spark.ml").setLevel(Level.INFO)
    Logger.getLogger("org.apache.spark.ml.tree.impl.RandomForest").setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark.ml.tree.impl.CompletelyRandomForestImpl").setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark.ml.classification.CompletelyRandomForestClassifier").setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark.ml.classification.RandomForestClassifier").setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark.ml.util.Instrumentation").setLevel(Level.ERROR)

    val logger = Logger.getLogger(this.getClass)
    logger.info(s"debugLevel: ${param.debugLevel}")
    logger.info(s"Total Cores is $parallelism")

    spark.sparkContext.setCheckpointDir(param.checkpointDir)
    val output = param.model

    def getParallelism: Int = param.parallelism match {
      case p if p > 0 => param.parallelism
      case n if n < 0 => -1
      case _ => parallelism
    }

    // generate rsp blocks and select g blocks
    val srcFile = param.trainFile
    val rspBlock = param.rspBlockNum
    val g = param.g
    val r = param.subRFNum

    //    val randomDouble = Random.nextDouble()
    val randomDouble = 0.9051215227917841
    val tempTrainDir = s"linyigang/data/temp/intermediate/${param.dataset}/rspTrain/${param.rspBlockNum}/${randomDouble}"
    val dstTrainDir = s"linyigang/data/temp/dst/${param.dataset}/rspTrain/${param.rspBlockNum}/${randomDouble}"
    val fullTrain = s"linyigang/data/temp/full/${param.dataset}/allTrain"
    val fullTest = s"linyigang/data/temp/full/${param.dataset}/allTest"

    // 生成rsp块，如果rsp块已经分好了，就把下面这几行注释掉即可
    val rddSource = spark.sparkContext.textFile(srcFile)
    val Array(train, test) = rddSource.randomSplit(Array(0.7, 0.3))


    val rspTrain = train.toRSP(rspBlock)
    //    val rspTest = test.toRSP(rspBlock)

    val configuration: Configuration = new Configuration()
    val fileSystem: FileSystem = FileSystem.newInstance(configuration)


    val tempTrainDirPath = new Path(tempTrainDir)
    if (fileSystem.exists(tempTrainDirPath)) fileSystem.delete(tempTrainDirPath, true)
    val fullTrainPath = new Path(fullTrain)
    if (fileSystem.exists(fullTrainPath)) fileSystem.delete(fullTrainPath, true)
    val fullTestPath = new Path(fullTest)
    if (fileSystem.exists(fullTestPath)) fileSystem.delete(fullTestPath, true)

    train.coalesce(1).saveAsTextFile(fullTrain)
    test.coalesce(1).saveAsTextFile(fullTest)

    val dstTrainDirPath = new Path(dstTrainDir)
    println(s"fileSystem.exists(dstTrainDirPath):${fileSystem.exists(dstTrainDirPath)}")
    if (!fileSystem.exists(dstTrainDirPath)) {
      rspTrain.saveAsTextFile(tempTrainDir)
      copyDirToLocal(spark, fileSystem, tempTrainDir, dstTrainDir)

    }
    copyDirToLocal(spark, fileSystem, fullTrain, fullTrain + "1")
    copyDirToLocal(spark, fileSystem, fullTest, fullTest + "1")

    fileSystem.close()

    val selectedIndexesG = Random.shuffle(0 to rspBlock - 1).take(g).toArray

    val selectedRSP = Array.ofDim[Dataset[_]](g)

    println("selectedIndexes outer: " + selectedIndexesG.mkString(","))
    for (i <- 0 to g - 1) {
      println(f"${dstTrainDir}/part-${selectedIndexesG(i)}%05d.csv")
      selectedRSP(i) = load_data(spark, param,
        f"${dstTrainDir}/part-${selectedIndexesG(i)}%05d.csv")
      println(s"selectedRSP:(${selectedRSP(i).count()},${selectedRSP(i).first()})")
    }
    val classNum = param.dataset match {
      case "uci_adult" => 2
      case "covertype" => 7
      case "watch_acc" => 18
      case "susy" => 2
      case "higgs" => 2
    }
    val fullTestData = load_data(spark, param, s"${fullTest}1/part-00000.csv")
    val fullTrainData = load_data(spark, param, s"${fullTrain}1/part-00000.csv")

    // set the parsed parameters to RSPGCForestClassifier and then train it
    val rspGCForest = new RSPGCForestClassifier()
      .setModelPath(param.model)
      .setDataSize(param.dataSize)
      .setDataStyle(param.dataStyle)
      .setMultiScanWindow(param.multiScanWindow)
      .setRFNum(param.rfNum)
      .setCRFNum(param.crfNum)
      .setCascadeForestTreeNum(param.cascadeForestTreeNum)
      .setScanForestTreeNum(param.scanForestTreeNum)
      .setMaxIteration(param.maxIteration)
      .setMaxDepth(param.maxDepth)
      .setMaxBins(param.maxBins)
      .setMinInfoGain(param.minInfoGain)
      .setMaxMemoryInMB(param.maxMemoryInMB)
      .setCacheNodeId(param.cacheNodeId)
      .setScanForestMinInstancesPerNode(param.scanMinInsPerNode)
      .setCascadeForestMinInstancesPerNode(param.cascadeMinInsPerNode)
      .setFeatureSubsetStrategy(param.featureSubsetStrategy)
      .setCrf_featureSubsetStrategy(param.crf_featureSubsetStrategy)
      .setEarlyStoppingRounds(param.earlyStoppingRounds)
      .setIDebug(param.idebug)
      .setSubRFNum(param.subRFNum)
      .setLambda(param.lambda)
      .setNumClasses(classNum)
      .setNewFeatures(param.newFeatures)
      .setDataset(param.dataset)

    val model = rspGCForest.train(selectedRSP, fullTestData)
    spark.sparkContext.getPersistentRDDs.foreach(truple => {
      truple._2.unpersist()
    })

    val totalTime = (System.currentTimeMillis() - stime) / 1000.0
    logger.info(s"Total time for RSPGCForest Application: $totalTime")

    Thread.sleep(20000)
    spark.stop()
  }


  def load_data(spark: SparkSession, param: TrainParams, trainFile: String, testFile: String = ""): Dataset[_] = {
    //     read dataset and split it into training and test sets
    val train = param.dataset match {
      case "uci_adult" => {
        val train = new UCI_adult().load_data(spark, trainFile, param.featuresFile, 1,
          spark.sparkContext.defaultParallelism)
        //        val test = new UCI_adult().load_data(spark, testFile, param.featuresFile, 1,
        //          spark.sparkContext.defaultParallelism)
        // Cast LabelCol to DoubleType and keep the metadata.
        //        (train, test)
        train
      }
      case "covertype" => {
        val train = new Covertype().load_data(spark, trainFile, param.featuresFile, 1,
          spark.sparkContext.defaultParallelism)
        //        val test = new Covertype().load_data(spark, testFile, param.featuresFile, 1,
        //          spark.sparkContext.defaultParallelism)
        //        (train, test)
        train
      }
      case "watch_acc" => {
        val train = new WatchAcc().load_data(spark, trainFile, param.featuresFile, 1,
          spark.sparkContext.defaultParallelism)
        //        val test = new WatchAcc().load_data(spark, testFile, param.featuresFile, 1,
        //          spark.sparkContext.defaultParallelism)
        //        (train, test)
        train
      }
      case "susy" => {
        val train = new SUSY().load_data(spark, trainFile, param.featuresFile, 1,
          spark.sparkContext.defaultParallelism)
        //        val test = new SUSY().load_data(spark, testFile, param.featuresFile, 1,
        //          spark.sparkContext.defaultParallelism)
        //        (train, test)
        train
      }
      case "higgs" => {
        val train = new HIGGS().load_data(spark, trainFile, param.featuresFile, 1,
          spark.sparkContext.defaultParallelism)
        //        val test = new HIGGS().load_data(spark, testFile, param.featuresFile, 1,
        //          spark.sparkContext.defaultParallelism)
        //        (train, test)
        train
      }
    }
    train
  }

  def transport(inputStream: InputStream, outputStream: OutputStream): Unit = {
    val buffer = new Array[Byte](4096)
    var len = inputStream.read(buffer)
    while (len >= 0) {
      //     while (len != -1) {
      outputStream.write(buffer, 0, len)
      len = inputStream.read(buffer)
    }


    outputStream.flush()
    inputStream.close()
    outputStream.close()
  }

  //hdfs：hdfs文件系统，hdfsSource：hdfs文件的路径，localTarget：本地文件的路径
  def copyFileToLocal(hdfs: FileSystem, hdfsSource: String, localTarget: String): Unit = {
    val conf = hdfs.getConf
    FileUtil.copy(hdfs, new Path(hdfsSource), hdfs, new Path(localTarget), true, conf)
  }

  // 给出hdfsDirectory，hdfs的文件夹目录，返回需要文件夹中以part开头的文件名 List[String]
  def getHDFSFiles(hdfs: FileSystem, hdfsDirectory: String): Array[String] = {
    val fsPath: Path = new Path(hdfsDirectory)
    val iterator = hdfs.listFiles(fsPath, false)
    val list = new ListBuffer[String]
    while (iterator.hasNext) {
      val pathStatus = iterator.next()
      val hdfsPath = pathStatus.getPath
      val fileName = hdfsPath.getName
      list += fileName // list.append(fileName)
      //         println(fileName)
    }
    list.toArray.filter(_.startsWith("part"))
  }

  //根据hdfs文件的String列表，处理成csv文件，并将csv文件下载到本地
  def copyDirToLocal(spark: SparkSession, hdfs: FileSystem, hdfsSourceDir: String, localTargetDir: String): Unit = {
    val dstDir = new File(localTargetDir)
    if (!dstDir.exists()) {
      dstDir.mkdirs()
    }
    val srcFileList = getHDFSFiles(hdfs, hdfsSourceDir)
    //    srcFileList.foreach(x => println("==>" + x))
    srcFileList.foreach(srcFile => {

      val df = spark.read.format("csv").option("header", "false").option("inferSchema", "true").load(f"$hdfsSourceDir/$srcFile")
      df.repartition(1).write.format("csv").mode("overwrite").option("header", "false").save(s"$hdfsSourceDir/csv/$srcFile")
      //         假设文件夹下只有一个文件以part开头，如 part-00000-65e93f2e-038d-41e4-b03d-d2d069daff19-c000.csv
      val filename = getHDFSFiles(hdfs, s"$hdfsSourceDir/csv/$srcFile")(0)
      copyFileToLocal(hdfs, s"$hdfsSourceDir/csv/$srcFile/$filename", s"$localTargetDir/$srcFile.csv")
      println(s"copy file from $hdfsSourceDir/csv/$srcFile/$filename ==> $localTargetDir/$srcFile.csv ")
    })
  }
}

