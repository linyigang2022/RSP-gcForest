/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.evaluation

class F1Score(
                 private var TP: Array[Double],
                 private var FP: Array[Double],
                 private var FN: Array[Double]
             )
  extends Metric {

  def getF1Score: Double = {
    val macroPrecision = (this.TP, this.FP).zipped.map { (tp, fp) =>
      tp / (if (tp + fp <= 0) 1.0 else tp + fp)
    }.foldLeft(0.0) (_+_)/this.TP.length
    val macroRecall = (this.TP, this.FN).zipped.map { (tp, fn) =>
      tp / (if (tp + fn <= 0) 1.0 else tp + fn)
    }.foldLeft(0.0)(_ + _) / this.TP.length
    (2 * macroPrecision * macroRecall) / (macroPrecision + macroRecall)
  }

  def +(that: F1Score): F1Score = {
    this.TP = (this.TP, that.TP).zipped.map(_ + _)
    this.FP = (this.FP, that.FP).zipped.map(_ + _)
    this.FN = (this.FN, that.FN).zipped.map(_ + _)
    this
  }


  override def +(that: Metric): F1Score = {
    require(that.isInstanceOf[F1Score], "Not an Precision Object")
    this.TP = (this.TP, that.asInstanceOf[F1Score].TP).zipped.map(_ + _)
    this.FP = (this.FP, that.asInstanceOf[F1Score].FP).zipped.map(_ + _)
    this.FN = (this.FN, that.asInstanceOf[F1Score].FN).zipped.map(_ + _)
    this
  }

  override def /(value: Double): F1Score = {
    this.TP = this.TP.map(_ / value)
    this
  }

  def div(value: Double): F1Score = {
    this.TP = this.TP.map(_ / value)
    this.FP = this.FP.map(_ / value)
    this.FN = this.FN.map(_ / value)
    this
  }

  override def toString: String = {
    s"F1-Score(TP: $TP, FP: $FP, $FN: $FN) = %.3f%%)".format(getF1Score * 100.0)
  }

  def equalsTo(obj: F1Score): Boolean = {
    (this.TP == obj.TP) && (this.FP == obj.FP) && (this.FN == obj.FN)
  }

  def reset(): F1Score = {
    this.TP = Array.ofDim[Double](this.TP.length)
    this.FP = Array.ofDim[Double](this.FP.length)
    this.FN = Array.ofDim[Double](this.FN.length)
    this
  }
}
