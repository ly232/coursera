package nodescala

import scala.language.postfixOps
import scala.util.{Try, Success, Failure}
import scala.collection._
import scala.concurrent._
import ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.async.Async.{async, await}
import org.scalatest._
import NodeScala._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class NodeScalaSuite extends FunSuite {

  test("A Future should always be completed") {
    val always = Future.always(517)

    assert(Await.result(always, 0 nanos) == 517)
  }
  test("A Future should never be completed") {
    val never = Future.never[Int]

    try {
      Await.result(never, 1 second)
      assert(false)
    } catch {
      case t: TimeoutException => // ok!
    }
  }
  test("A list of futures should complete with a future of list") {
    val all = Future.all(List(Future(1), Future(2), Future(3)))
    assert(Await.result(all, 100 seconds) == List(1, 2, 3))
  }
  test("A list of futures should fail if any individual future fails") {
    val all = Future.all(
        List(Future(1), Future(2), Future(throw new Exception)))
    all onComplete {
      case Success(_) => assert(false)
      case Failure(_) => assert(true)
    }
  }
  test("A list of futures should complete with value") {
    val any = Future.any(
        List(Future(1), Future.never[Exception]))
    assert(Await.result(any, 100 seconds) == 1)
  }
  test("A list of futures should complete with exception") {
    val any = Future.any(
        List(Future.never[Int], Future(throw new Exception)))
    try {
      Await.result(any, 100 seconds)
      assert(false, "an exception is expected.")
    } catch {
      case t: Exception => // ok!
    }
  }
  test("Delayed future should be completed after delay") {
    val delay = Future.delay(5 millis)
    delay onComplete {
      case Success(_) => assert(true)
      case Failure(_) => assert(false, "delayed promise failed to complete.")
    }
    try {
      Await.result(delay, 4 millis)
      assert(false, "delayed future copmleted too soon.")
    } catch {
      case t: TimeoutException => {
        Await.result(delay, 100 seconds)
      }
    }
  }
  test("now should return future's result now") {
    val always = Future.always(123)
    assert(always.now == 123)
    val never = Future.never[Int]
    try {
      never.now
      assert(false, "never should not complete")
    } catch {
      case t: NoSuchElementException => assert(true)
      case _: Throwable => assert(false, "wrong exception thrown")
    }
  }
  test("continueWith should continue operation with succeeded future") {
    val f = Future(1)
    val res = f.continueWith(x => Await.result(x, 100 seconds) + 1)
    assert(Await.result(res, 100 seconds) == 2)
  }
  test("continue should continue operation with succeeded future") {
    val f = Future(1)
    val res = f.continue(x => x match {
      case Success(v) => v + 1
      case Failure(e) => assert(false, "future failed to complete")
    })
    assert(Await.result(res, 100 seconds) == 2)
  }
  test("should cancel computation eventually") {
    val p = Promise[Int]()
    val working = Future.run() {
      ct => Future {
        while (ct.nonCancelled) {
          println("working")
        }
        println("done")
        p.success(1)
      }
    }
    Future.delay(1 seconds) onSuccess {
      case _ => working.unsubscribe()
    }
    assert(Await.result(p.future, Duration.Inf) == 1)
  }
  test("CancellationTokenSource should allow stopping the computation") {
    val cts = CancellationTokenSource()
    val ct = cts.cancellationToken
    val p = Promise[String]()

    async {
      while (ct.nonCancelled) {
        // do work
      }

      p.success("done")
    }

    cts.unsubscribe()
    assert(Await.result(p.future, 1 second) == "done")
  }

  
  
  class DummyExchange(val request: Request) extends Exchange {
    @volatile var response = ""
    val loaded = Promise[String]()
    def write(s: String) {
      response += s
    }
    def close() {
      loaded.success(response)
    }
  }

  class DummyListener(val port: Int, val relativePath: String) extends NodeScala.Listener {
    self =>

    @volatile private var started = false
    var handler: Exchange => Unit = null

    def createContext(h: Exchange => Unit) = this.synchronized {
      assert(started, "is server started?")
      handler = h
    }

    def removeContext() = this.synchronized {
      assert(started, "is server started?")
      handler = null
    }

    def start() = self.synchronized {
      started = true
      new Subscription {
        def unsubscribe() = self.synchronized {
          started = false
        }
      }
    }

    def emit(req: Request) = {
      val exchange = new DummyExchange(req)
      if (handler != null) handler(exchange)
      exchange
    }
  }

  class DummyServer(val port: Int) extends NodeScala {
    self =>
    val listeners = mutable.Map[String, DummyListener]()

    def createListener(relativePath: String) = {
      val l = new DummyListener(port, relativePath)
      listeners(relativePath) = l
      l
    }

    def emit(relativePath: String, req: Request) = this.synchronized {
      val l = listeners(relativePath)
      l.emit(req)
    }
  }
  test("Server should serve requests") {
    val dummy = new DummyServer(8191)
    val dummySubscription = dummy.start("/testDir") {
      request => for (kv <- request.iterator) yield (kv + "\n").toString
    }

    // wait until server is really installed
    Thread.sleep(500)

    def test(req: Request) {
      val webpage = dummy.emit("/testDir", req)
      val content = Await.result(webpage.loaded.future, 1 second)
      val expected = (for (kv <- req.iterator) yield (kv + "\n").toString).mkString
      assert(content == expected, s"'$content' vs. '$expected'")
    }

    test(immutable.Map("StrangeRequest" -> List("Does it work?")))
    test(immutable.Map("StrangeRequest" -> List("It works!")))
    test(immutable.Map("WorksForThree" -> List("Always works. Trust me.")))

    dummySubscription.unsubscribe()
  }

}




