package calculator

sealed abstract class Expr
final case class Literal(v: Double) extends Expr
final case class Ref(name: String) extends Expr
final case class Plus(a: Expr, b: Expr) extends Expr
final case class Minus(a: Expr, b: Expr) extends Expr
final case class Times(a: Expr, b: Expr) extends Expr
final case class Divide(a: Expr, b: Expr) extends Expr

object Calculator {
  def computeValues(
      namedExpressions: Map[String, Signal[Expr]]): Map[String, Signal[Double]] = {
    namedExpressions.mapValues(sig => Signal({eval(sig(), namedExpressions)}))
  }

  def eval(expr: Expr, references: Map[String, Signal[Expr]]): Double = {
    var pendingRefs: Set[String] = Set()
    def evalHelper(expr: Expr, references: Map[String, Signal[Expr]]): Double = {
      expr match {
        case Literal(v) => v
        case Ref(name) => {
          if (pendingRefs.contains(name)) Double.NaN
          else {
            pendingRefs += name
            evalHelper(getReferenceExpr(name, references), references)
          }
        }
        case Plus(a, b) =>
          evalHelper(a, references) + evalHelper(b, references)
        case Minus(a, b) =>
          evalHelper(a, references) - evalHelper(b, references)
        case Times(a, b) =>
          evalHelper(a, references) * evalHelper(b, references)
        case Divide(a, b) =>
          evalHelper(a, references) / evalHelper(b, references)
        case _ => Double.NaN
      }
    }
    evalHelper(expr, references)
  }

  /** Get the Expr for a referenced variables.
   *  If the variable is not known, returns a literal NaN.
   */
  private def getReferenceExpr(name: String,
      references: Map[String, Signal[Expr]]) = {
    references.get(name).fold[Expr] {
      Literal(Double.NaN)
    } { exprSignal =>
      exprSignal()
    }
  }
}
