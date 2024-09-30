package edu.coursera.concurrent;

import static edu.rice.pcdp.PCDP.finish;

import edu.rice.pcdp.Actor;
import java.util.List;
import java.util.ArrayList;

/**
 * An actor-based implementation of the Sieve of Eratosthenes.
 *
 * TODO Fill in the empty SieveActorActor actor class below and use it from
 * countPrimes to determin the number of primes <= limit.
 */
public final class SieveActor extends Sieve {
    /**
     * {@inheritDoc}
     *
     * TODO Use the SieveActorActor class to calculate the number of primes <=
     * limit in parallel. You might consider how you can model the Sieve of
     * Eratosthenes as a pipeline of actors, each corresponding to a single
     * prime number.
     */
    @Override
    public int countPrimes(final int limit) {
        // final List<Integer> localPrimes = new ArrayList<Integer>();
        // localPrimes.add(2);
        final SieveActorActor actor = new SieveActorActor(2, limit);
        finish(() -> {
            for (int i = 3; i < limit; ++i) {
                actor.send(i);
            }
            actor.send(0);
        });

        int cnt = 0;
        SieveActorActor currActor = actor;
        while (currActor != null) {
            cnt += 1;
            currActor = currActor.getNextActor();
        }
        return cnt;
    }

    /**
     * An actor class that helps implement the Sieve of Eratosthenes in
     * parallel.
     */
    public static final class SieveActorActor extends Actor {
        private final int seedPrime;
        private final int limit;
        private SieveActorActor nextActor = null;

        public SieveActorActor(int seedPrime, int limit) {
            this.seedPrime = seedPrime;
            this.limit = limit;
        }

        public SieveActorActor getNextActor() {
            return this.nextActor;
        }

        /**
         * Process a single message sent to this actor.
         *
         * TODO complete this method.
         *
         * @param msg Received message
         */
        @Override
        public void process(final Object msg) {
            final int candidate = (int) msg;
            if (candidate == 0 || candidate > this.limit) {
                // Termination case.
                if (this.nextActor != null) {
                    this.nextActor.send(candidate);
                }
                return;
            }
            if (candidate % this.seedPrime == 0) {
                return;
            } else if (this.nextActor == null) {
                this.nextActor = new SieveActorActor(candidate, limit);
            } else {
                this.nextActor.send(candidate);
            }
        }
    }
}
