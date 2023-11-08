/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.datasets

import org.apache.log4j.Logger
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

//one hot
//1 2 3
//[1 0 0]
//[0 1 0]
//[0 0 1]
class HIGGS extends BaseDatasets {
  /**
   * Load HIGGS data, by sparkSession, phase(or file path) and cate_as_onehot
   *
   * @param spark          SparkSession to load
   * @param phase          which kind of data to load, "train" or "test", or provide file path directly
   * @param cate_as_onehot convert categorical data to one-hot format
   * @return loaded DataFrame
   */
  def load_data(spark: SparkSession,
                phase: String,
                featuresPath: String,
                cate_as_onehot: Int,
                repar: Int = 0): DataFrame = {

    val data_path =
      if (phase == "train") "linyigang/data/uci_adult/adult.data"
      else if (phase == "test") "linyigang/data/uci_adult/adult.test"
      else phase

    val logger = Logger.getLogger(this.getClass)
    logger.info(data_path)

    val raw = spark.read.text(data_path)
    val features_path = if (featuresPath == "") "linyigang/data/uci_adult/features" else featuresPath

    // new featureParser, the featureParser can convert discrete features to one-hot encoding, and continuous features do not require special handling.
    val fts_file = spark.read.text(features_path)
    val f_parsers = fts_file.rdd.filter(row => row.length > 0).map { row =>
      val line = row.getAs[String]("value")
      new FeatureParser(line)
    }
    val total_dims = f_parsers.map { fts =>
      fts.get_fdim()
    }.reduce((l, r) => l + r)

    val f_parsers_array = f_parsers.collect()

    // process the dataset
    val dataRDD = raw.rdd.filter(row => row.mkString.length > 1 && !row.mkString.startsWith("|"))
      .zipWithIndex.map { case (row, idx) =>
      val line = row.getAs[String]("value")
      val splits = line.split(",")
      require(splits.length == 29 || splits.length == 31, s"row $idx: $line has no 29 features, length: ${splits.length}")
      val (label, data) = if (splits.length == 29) {
        val label = splits(0).toDouble
        val data = splits.drop(1).zipWithIndex.map { case (feature, indx) =>
          f_parsers_array(indx).get_data(feature.trim)
        }.reduce((l, r) => l ++ r)
        (label, data)
      } else {
        val label = splits(30).toDouble
        val data = splits.dropRight(1).zipWithIndex.map { case (feature, indx) =>
          f_parsers_array(indx).get_data(feature.trim)
        }.reduce((l, r) => l ++ r)
        (label, data)
      }
      require(data.length == total_dims,
        "Total dims %d not equal to data.length %d".format(total_dims, data.length))
      Row.fromSeq(Seq[Any](label, data, idx))
    }
    val repartitioned = if (repar > 0) dataRDD.repartition(repar) else dataRDD
    val schema = new StructType()
      .add(StructField("label", DoubleType))
      .add(StructField("features", ArrayType(DoubleType)))
      .add(StructField("instance", LongType))
    val arr2vec = udf { (features: Seq[Double]) => new DenseVector(features.toArray) }
    spark.createDataFrame(repartitioned, schema)
      .withColumn("features", arr2vec(col("features")))
  }
}