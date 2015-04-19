package quickcheck

import common._
import scala.math._

import org.scalacheck._
import Arbitrary._
import Gen._
import Prop._

abstract class QuickCheckHeap extends Properties("Heap") with IntHeap {

  property("findMin should return a if a is the only element in heap.") = forAll { a: Int =>
    val h = insert(a, empty)
    findMin(h) == a
  }

  /*
   * If you insert any two elements into an empty heap,
   * finding the minimum of the resulting heap should get the
   * smallest of the two elements back.
   * */
  property("findMin should return min element in heap.") = forAll { a: Int =>
    val h1 = insert(a, empty)
    val h2 = insert(a + 1, h1)
    findMin(h2) == (if (Int.MaxValue == a) a + 1 else a)
  }

  /*
   * If you insert an element into an empty heap, then delete the minimum,
   * the resulting heap should be empty.
   * */
  property("deleteMin should make heap empty if heap only has 1 element") =
    forAll { a: Int =>
      val h1 = insert(a, empty)
      val h2 = deleteMin(h1)
      h2 == empty
    }

  /*
   * Finding a minimum of the melding of any two heaps should
   * return a minimum of one or the other.
   * */
  property("find min element in merged heap") =
    forAll(genHeap, genHeap)((h1, h2) =>
      findMin(meld(h1, h2)) == min(findMin(h1), findMin(h2)))

  /*
   * Given any heap, you should get a sorted sequence of
   * elements when continually finding and deleting minima.
   * (Hint: recursion and helper functions are your friends.)
   * */
  property("sorted distinct") = forAll { _: Int =>
    val h1 = insert(1, empty)
    val h2 = insert(3, h1)
    val h3 = insert(2, h2)
    def validate(h: H): Boolean = {
      if (isEmpty(h)) true
      else {
        val nh = deleteMin(h)
        validate(nh) && (
          if (isEmpty(nh)) true else findMin(nh) > findMin(h))
      }
    }
    validate(h3)
  }

  lazy val genHeap: Gen[H] = for {
    v <- arbitrary[Int]
    h <- oneOf(const(empty), genHeap)
  } yield insert(v, h)

  implicit lazy val arbHeap: Arbitrary[H] = Arbitrary(genHeap)

}
