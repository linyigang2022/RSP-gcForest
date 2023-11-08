/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.datasets

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
class BaseDatasets extends Logging{
  def load_data1(spark: SparkSession,
                phase: String,
                repar: Int = 0): DataFrame = {

    val data_path =
      if (phase == "train") "linyigang/data/uci_adult/adult.data"
      else if (phase == "test") "linyigang/data/uci_adult/adult.test"
      else phase

//    logInfo(data_path)
    val raw = spark.read.text(data_path)

    val dataRDD = raw.rdd.filter(row => row.mkString.length > 1 && !row.mkString.startsWith("|"))
      .zipWithIndex.map { case (row, idx) =>
      val line = row.getAs[String]("value")
      val splits = line.split(",")
//      require(splits.length == 24, s"row $idx: $line has no 23 features, length: ${splits.length}")
      val (label, data, instance) = {
        val label = splits.last.trim.toDouble
        val instance = splits(0).trim.toLong
        val data = splits.dropRight(1).drop(1).zipWithIndex.map { case (feature, indx) =>
          Array[Double](feature.trim.toDouble)
        }.reduce((l, r) => l ++ r)
        (label, data, instance)
        }
      Row.fromSeq(Seq[Any](label, data, instance))
    }
//    logInfo(s"处理后的数据") //${dataRDD.first().mkString("_")}")
//    dataRDD.take(5).foreach(x =>
//      logInfo(x(1).asInstanceOf[Array[Double]].mkString(",")))

    val repartitioned = if (repar > 0) dataRDD.repartition(repar) else dataRDD
    val schema = new StructType()
      .add(StructField("label", DoubleType))
      .add(StructField("features", ArrayType(DoubleType)))
      .add(StructField("instance", LongType))

    val arr2vec = udf { (features: Seq[Double]) => new DenseVector(features.toArray) }
    spark.createDataFrame(repartitioned, schema)
      .withColumn("features", arr2vec(col("features")))
  }

  def load_data2(spark: SparkSession,
                 phase: String,
                 repar: Int = 0): DataFrame = {

    val data_path =
      if (phase == "train") "linyigang/data/uci_adult/adult.data"
      else if (phase == "test") "linyigang/data/uci_adult/adult.test"
      else phase

    //    logInfo(data_path)
    val raw = spark.read.text(data_path)

    val dataRDD = raw.rdd.filter(row => row.mkString.length > 1 && !row.mkString.startsWith("|"))
      .zipWithIndex.map { case (row, idx) =>
      val line = row.getAs[String]("value")
      val splits = line.split(",")
      //      require(splits.length == 24, s"row $idx: $line has no 23 features, length: ${splits.length}")
      val (label, data, instance, rsp) = {
        val len = splits.length
        val label = splits(len-2).trim.toDouble
        val instance = splits(0).trim.toLong
        val rsp = splits(len-1).trim.toLong
        val data = splits.dropRight(2).drop(1).zipWithIndex.map { case (feature, indx) =>
          Array[Double](feature.trim.toDouble)
        }.reduce((l, r) => l ++ r)
        (label, data, instance, rsp)
      }
      Row.fromSeq(Seq[Any](label, data, instance, rsp))
    }
    //    logInfo(s"处理后的数据") //${dataRDD.first().mkString("_")}")
    //    dataRDD.take(5).foreach(x =>
    //      logInfo(x(1).asInstanceOf[Array[Double]].mkString(",")))

    val repartitioned = if (repar > 0) dataRDD.repartition(repar) else dataRDD
    val schema = new StructType()
      .add(StructField("label", DoubleType))
      .add(StructField("features", ArrayType(DoubleType)))
      .add(StructField("instance", LongType))
      .add(StructField("rsp", LongType))

    val arr2vec = udf { (features: Seq[Double]) => new DenseVector(features.toArray) }
    spark.createDataFrame(repartitioned, schema)
      .withColumn("features", arr2vec(col("features")))
  }

}
class FeatureParser(row: String) extends Serializable  {
  private val desc = row.trim
  private val f_type = if (desc == "C") "number" else "categorical"
  private val name_to_len = if (f_type == "categorical") {
    val f_names = Array("?") ++ desc.trim.split(",").map(str => str.trim)
    f_names.zipWithIndex.map { case(cate_name, idx) =>
      cate_name -> idx
    }.toMap
  } else Map[String, Int]()

  def get_double(f_data: String): Double = {
    if (f_type == "number") f_data.trim.toDouble
    else name_to_len.getOrElse(f_data.trim, 0).toDouble
  }

  def get_data(f_data: String): Array[Double] = {
    if (f_type == "number") Array[Double](f_data.trim.toDouble)
    else {
      val data = Array.fill[Double](name_to_len.size)(0f)
      data(name_to_len.getOrElse(f_data.trim, 0)) = 1f
      data
    }
  }

  def get_fdim(): Int = {
    if (f_type == "number") 1 else name_to_len.size
  }
}