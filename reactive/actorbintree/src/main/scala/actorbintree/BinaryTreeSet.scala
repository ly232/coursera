/**
 * Copyright (C) 2009-2013 Typesafe Inc. <http://www.typesafe.com>
 */
package actorbintree

import akka.actor._

object BinaryTreeSet {

  trait Operation {
    def requester: ActorRef
    def id: Int
    def elem: Int
  }

  trait OperationReply {
    def id: Int
  }

  /** Request with identifier `id` to insert an element `elem` into the tree.
    * The actor at reference `requester` should be notified when this operation
    * is completed.
    */
  case class Insert(requester: ActorRef, id: Int, elem: Int) extends Operation

  /** Request with identifier `id` to check whether an element `elem` is present
    * in the tree. The actor at reference `requester` should be notified when
    * this operation is completed.
    */
  case class Contains(requester: ActorRef, id: Int, elem: Int) extends Operation

  /** Request with identifier `id` to remove the element `elem` from the tree.
    * The actor at reference `requester` should be notified when this operation
    * is completed.
    */
  case class Remove(requester: ActorRef, id: Int, elem: Int) extends Operation

  /** Request to perform garbage collection*/
  case object GC

  /** Holds the answer to the Contains request with identifier `id`.
    * `result` is true if and only if the element is present in the tree.
    */
  case class ContainsResult(id: Int, result: Boolean) extends OperationReply
  
  /** Message to signal successful completion of an insert or remove operation. */
  case class OperationFinished(id: Int) extends OperationReply

}


class BinaryTreeSet extends Actor with Stash {
  import BinaryTreeSet._
  import BinaryTreeNode._

  def createRoot: ActorRef = context.actorOf(BinaryTreeNode.props(0, initiallyRemoved = true))

  var root = createRoot

  // optional
  def receive = normal

  // optional
  /** Accepts `Operation` and `GC` messages. */
  val normal: Receive = {
    case GC => {
      val newRoot = createRoot
      root ! CopyTo(newRoot)
      context.become(garbageCollecting(newRoot))
    }
    case op => root ! op
  }

  // optional
  /** Handles messages while garbage collection is performed.
    * `newRoot` is the root of the new binary tree where we want to copy
    * all non-removed elements into.
    */
  def garbageCollecting(newRoot: ActorRef): Receive = {
    case CopyFinished => {
      root ! PoisonPill
      root = newRoot
      context.become(normal)
      unstashAll()
    }
    case op => stash()
  }

}

object BinaryTreeNode {
  trait Position

  case object Left extends Position
  case object Right extends Position

  case class CopyTo(treeNode: ActorRef)
  case object CopyFinished

  def props(elem: Int, initiallyRemoved: Boolean) = Props(classOf[BinaryTreeNode],  elem, initiallyRemoved)
}

class BinaryTreeNode(val elem: Int, initiallyRemoved: Boolean) extends Actor {
  import BinaryTreeNode._
  import BinaryTreeSet._

  var subtrees = Map[Position, ActorRef]()
  var removed = initiallyRemoved

  // optional
  def receive = normal

  def decideLeftRight(elem: Int) = if (elem  < this.elem) Left else Right

  // optional
  /** Handles `Operation` messages and `CopyTo` requests. */
  val normal: Receive = {
    case Insert(requester, id, elem) => {
      subtrees.get(decideLeftRight(elem)) match {
        case Some(child) => child ! Insert(requester, id, elem)
        case None => {
          subtrees = subtrees + (decideLeftRight(elem) ->
            context.actorOf(BinaryTreeNode.props(elem, initiallyRemoved = false)))
          requester ! OperationFinished(id)
        }
      }
    }
    case Contains(requester, id, elem) => {
      if (elem == this.elem) requester ! ContainsResult(id, !removed)
      else subtrees.get(decideLeftRight(elem)) match {
        case Some(child) => child ! Contains(requester, id, elem)
        case None => requester ! ContainsResult(id, false)
      }
    } 
    case Remove(requester, id, elem) => {
      if (elem == this.elem) {
        removed = true;
        requester ! OperationFinished(id)
      }
      else subtrees.get(decideLeftRight(elem)) match {
        case Some(child) => child ! Remove(requester, id, elem)
        case None => requester ! OperationFinished(id)
      }
    } 
    case CopyTo(treeNode) => {
      if (!removed) treeNode ! Insert(self, elem, elem)
      subtrees.foreach{
        case (_, child) => child ! CopyTo(treeNode)
      }
      context.become(copying(subtrees.values.toSet, removed))
    }
  }

  // optional
  def copying(expected: Set[ActorRef], insertConfirmed: Boolean): Receive = {
    if (expected.isEmpty && insertConfirmed) {
      context.parent ! CopyFinished
      normal
    } else {
      case OperationFinished(id) if (id == elem) => context.become(copying(expected, true))
      case CopyFinished => context.become(copying(expected - sender, insertConfirmed))
    }
  }

}
