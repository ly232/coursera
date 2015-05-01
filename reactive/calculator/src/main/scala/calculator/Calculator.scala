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
    def evalHelper(expr: Expr, references: Map[String, Signal[Expr]],
        pendingRefs: Set[String]): Double = {
      expr match {
        case Literal(v) => v
        case Ref(name) => {
          if (pendingRefs.contains(name)) Double.NaN
          else {
            evalHelper(getReferenceExpr(name, references), references,
                pendingRefs + name)
          }
        }
        case Plus(a, b) =>
          evalHelper(a, references, pendingRefs) +
          evalHelper(b, references, pendingRefs)
        case Minus(a, b) =>
          evalHelper(a, references, pendingRefs) -
          evalHelper(b, references, pendingRefs)
        case Times(a, b) =>
          evalHelper(a, references, pendingRefs) *
          evalHelper(b, references, pendingRefs)
        case Divide(a, b) =>
          evalHelper(a, references, pendingRefs) /
          evalHelper(b, references, pendingRefs)
        case _ => Double.NaN
      }
    }
    evalHelper(expr, references, Set[String]())
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
