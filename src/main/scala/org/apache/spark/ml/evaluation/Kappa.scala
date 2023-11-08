/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.evaluation

class Kappa(
                 private var TP: Array[Double],
                 private var FP: Array[Double],
                 private var FN: Array[Double]
             )
  extends Metric {

    def getKappa: Double = {
      val p0 = TP.sum / (TP.sum+FP.sum)
      val pe = (this.TP, this.FP, this.FN).zipped.map { (tp, fp, fn) =>
        (tp+fp)*(tp+fn)
      }.foldLeft(0.0)(_ + _) / ((TP.sum+FP.sum)*(TP.sum+FP.sum))
      (p0-pe)/(1-pe)
    }

    def +(that: Kappa): Kappa = {
      this.TP = (this.TP, that.TP).zipped.map(_ + _)
      this.FP = (this.FP, that.FP).zipped.map(_ + _)
      this.FN = (this.FN, that.FN).zipped.map(_ + _)
      this
    }


    override def +(that: Metric): Kappa = {
      require(that.isInstanceOf[Kappa], "Not an Precision Object")
      this.TP = (this.TP, that.asInstanceOf[Kappa].TP).zipped.map(_ + _)
      this.FP = (this.FP, that.asInstanceOf[Kappa].FP).zipped.map(_ + _)
      this.FN = (this.FN, that.asInstanceOf[Kappa].FN).zipped.map(_ + _)
      this
    }

    override def /(value: Double): Kappa = {
      this.TP = this.TP.map(_ / value)
      this
    }

    def div(value: Double): Kappa = {
      this.TP = this.TP.map(_ / value)
      this.FP = this.FP.map(_ / value)
      this.FN = this.FN.map(_ / value)
      this
    }

    override def toString: String = {
      s"F1-Score(TP: $TP, FP: $FP, $FN: $FN) = %.3f%%)".format(getKappa * 100.0)
    }

    def equalsTo(obj: Kappa): Boolean = {
      (this.TP == obj.TP) && (this.FP == obj.FP) && (this.FN == obj.FN)
    }

    def reset(): Kappa = {
      this.TP = Array.ofDim[Double](this.TP.length)
      this.FP = Array.ofDim[Double](this.FP.length)
      this.FN = Array.ofDim[Double](this.FN.length)
      this
    }
  }
