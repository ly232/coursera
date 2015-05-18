/**
 * Copyright (C) 2009-2013 Typesafe Inc. <http://www.typesafe.com>
 */
package actorbintree

import akka.actor._
import scala.collection.immutable.Queue

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
  var pendingQueue = Queue.empty[Operation]

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
  
  def createNode(elem: Int): ActorRef =
    context.actorOf(BinaryTreeNode.props(elem, initiallyRemoved = false))
    
  def decideLeftRight(elem: Int): Position =
    if (this.elem > elem) Left
    else if (this.elem < elem) Right
    else throw new Exception("Duplicate element: " + s"$elem")
  
  // Looks up children nodes and perform specified callbacks.
  def lookupAndDo(
      elem: Int, id: Int, someCallback: ActorRef => Unit,
      noneCallback: Position => Unit): Unit = {
      val pos = decideLeftRight(elem)
      subtrees.get(pos) match {
        case Some(child) => someCallback(child)
        case None => noneCallback(pos)
      }
    }

  // optional
  def receive = normal

  // optional
  /** Handles `Operation` messages and `CopyTo` requests. */
  val normal: Receive = {
    case Insert(requester, id, elem) => {
      if (elem == this.elem) {
        removed = false
        requester ! OperationFinished(id);
      }
      else lookupAndDo(
            elem, id, child => child ! Insert(requester, id, elem),
            pos => {
              subtrees = subtrees.updated(pos, createNode(elem));
            requester ! OperationFinished(id)})
    }
    case Contains(requester, id, elem) => {
      if (elem == this.elem) requester ! ContainsResult(id, !removed)
      else lookupAndDo(
          elem, id, child => child ! Contains(requester, id, elem),
          _ => requester ! ContainsResult(id, false))
    }
    case Remove(requester, id, elem) => {
      if (elem == this.elem) {
        removed = true
        requester ! OperationFinished(id);
      }
      else lookupAndDo(
            elem, id, child => child ! Remove(requester, id, elem),
            _ => requester ! OperationFinished(id))
    }
    case OperationFinished(id) => sender ! OperationFinished(id)
    case CopyTo(treeNode) => {
      if (!removed) treeNode ! Insert(self, elem, elem)
      subtrees foreach { case (_, child) => child ! CopyTo(treeNode) }
      context.become(copying(subtrees.values.toSet, removed))
    }
  }

  // optional
  /** `expected` is the set of ActorRefs whose replies we are waiting for,
    * `insertConfirmed` tracks whether the copy of this node to the new tree has been confirmed.
    */
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