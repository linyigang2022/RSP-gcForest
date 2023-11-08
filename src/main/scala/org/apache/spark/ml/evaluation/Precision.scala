/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.evaluation

class Precision(
                 private var TP: Array[Double],
                 private var FP: Array[Double])
  extends Metric {

  def getPrecision: Double = {
    val macroPrecision = (this.TP, this.FP).zipped.map { (tp, fp) =>
      tp / (if (tp + fp <= 0) 1.0 else tp + fp)
    }.foldLeft(0.0) (_+_)/this.TP.length
    macroPrecision
  }

  def +(that: Precision): Precision = {
    this.TP = (this.TP, that.TP).zipped.map(_ + _)
    this.FP = (this.FP, that.FP).zipped.map(_ + _)
    this
  }


  override def +(that: Metric): Precision = {
    require(that.isInstanceOf[Precision], "Not an Precision Object")
    this.TP = (this.TP, that.asInstanceOf[Precision].TP).zipped.map(_ + _)
    this.FP = (this.FP, that.asInstanceOf[Precision].FP).zipped.map(_ + _)
    this
  }

  override def /(value: Double): Precision = {
    this.TP = this.TP.map(_ / value)
    this
  }

  def div(value: Double): Precision = {
    this.TP = this.TP.map(_ / value)
    this.FP = this.FP.map(_ / value)
    this
  }

  override def toString: String = {
    s"Precision($TP /($TP + $FP) = %.3f%%)".format(getPrecision * 100.0)
  }

  def equalsTo(obj: Precision): Boolean = {
    (this.TP == obj.TP) && (this.FP == obj.FP)
  }

  def reset(): Precision = {
    this.TP = Array.ofDim[Double](this.TP.length)
    this.FP = Array.ofDim[Double](this.FP.length)
    this
  }
}
