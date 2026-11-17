"""
forkpool.ForkPool (Linux): wyniki i wyjątki jak w ProcessPoolExecutor, bez
semaforów (jak na AWS Lambda bez /dev/shm), śmierć procesu roboczego,
oraz pipeline.SharedPoolExecutor na tej puli (initializer, dalsze liczenie
w procesie głównym po awarii puli).
"""

import os
import signal
import sys
import time
import unittest
from concurrent.futures.process import BrokenProcessPool
from unittest import mock

from . import pipeline


def _square(value):
    return value * value


def _fail(value):
    raise ValueError(f"bad {value}")


def _exit(value):
    sys.exit(f"ERROR: {value}")


def _pid(_):
    return os.getpid()


def _die(_):
    os.kill(os.getpid(), signal.SIGKILL)


_INIT = {"calls": 0}


def _init(value):
    _INIT["calls"] += 1
    _INIT["value"] = value


def _state(_):
    return os.getpid(), _INIT["calls"], _INIT.get("value"), \
        dict(pipeline._OUTPUTS)


def _slow_square(value):
    time.sleep(0.05)
    return value * value


@unittest.skipUnless(sys.platform.startswith("linux"), "fork: Linux only")
class TestForkPool(unittest.TestCase):

    def test_results_exceptions_and_no_semaphores(self):
        from .forkpool import ForkPool
        import multiprocessing.synchronize as synchronize

        def no_semlock(*args, **kwargs):
            raise OSError("no /dev/shm")

        # Jak na AWS Lambda: każdy semafor multiprocessing kończy się błędem.
        with mock.patch.object(synchronize.SemLock, "__init__", no_semlock):
            self.assertFalse(pipeline.processes_available())
            pool = ForkPool(3)
        try:
            self.assertEqual(pool.map(_square, range(20)),
                             [v * v for v in range(20)])
            self.assertEqual(len(set(pool.map(_pid, range(30)))
                                 - {os.getpid()}), 3)
            with self.assertRaises(ValueError) as error:
                pool.submit(_fail, 7).result()
            self.assertIn("bad 7", str(error.exception))
            self.assertIn("_fail", str(error.exception.__cause__))
            with self.assertRaises(SystemExit):
                pool.submit(_exit, 1).result()
            # Pula działa dalej po wyjątkach w zadaniach.
            self.assertEqual(pool.submit(_square, 9).result(), 81)
        finally:
            pool.shutdown()
        for pid in pool.pids:
            with self.assertRaises(OSError):
                os.kill(pid, 0)

    def test_worker_death(self):
        from .forkpool import ForkPool
        pool = ForkPool(2)
        try:
            futures = [pool.submit(_slow_square, v) for v in range(6)]
            with self.assertRaises(BrokenProcessPool):
                pool.submit(_die, 0).result()
            for future in futures:
                try:
                    future.result()
                except BrokenProcessPool:
                    pass
            with self.assertRaises(BrokenProcessPool):
                pool.submit(_square, 2)
        finally:
            pool.shutdown()


@unittest.skipUnless(sys.platform.startswith("linux"), "fork: Linux only")
class TestSharedPoolExecutor(unittest.TestCase):

    def setUp(self):
        pipeline._load_modules()

    def tearDown(self):
        pipeline.set_shared_pool(None)

    def test_initializer_once_per_worker_and_outputs(self):
        from .forkpool import ForkPool
        pool = ForkPool(2)
        pipeline.set_shared_pool(pool)
        try:
            with mock.patch.dict(pipeline._OUTPUTS, {"photos": False}):
                executor = pipeline.SharedPoolExecutor(
                    max_workers=8, initializer=_init, initargs=("a",))
                states = executor.map(_state, range(12))
            for pid, calls, value, outputs in states:
                self.assertNotEqual(pid, os.getpid())
                self.assertEqual((calls, value), (1, "a"))
                self.assertFalse(outputs["photos"])
            # Nowy executor (następny etap) = nowy initializer.
            again = pipeline.SharedPoolExecutor(initializer=_init,
                                                initargs=("b",))
            for _, calls, value, _ in again.map(_state, range(6)):
                self.assertEqual((calls, value), (2, "b"))
            self.assertEqual(_INIT["calls"], 0)     # nie w tym procesie
        finally:
            pool.shutdown()

    def test_broken_pool_falls_back_inline(self):
        from .forkpool import ForkPool
        pool = ForkPool(2)
        pipeline.set_shared_pool(pool)
        try:
            executor = pipeline.SharedPoolExecutor(initializer=_init,
                                                   initargs=("c",))
            futures = [executor.submit(_slow_square, v) for v in range(8)]
            for pid in pool.pids:
                os.kill(pid, signal.SIGKILL)
            self.assertEqual([f.result() for f in futures],
                             [v * v for v in range(8)])
            # Kolejne zadania — od razu w tym procesie (initializer raz).
            self.assertEqual(executor.map(_square, range(4)), [0, 1, 4, 9])
            self.assertEqual(_INIT["calls"], 1)
            self.assertEqual(_INIT["value"], "c")
        finally:
            pool.shutdown()
            _INIT.clear()
            _INIT["calls"] = 0

    def test_without_pool_uses_fallback(self):
        pipeline.set_shared_pool(None)
        executor = pipeline.SharedPoolExecutor(max_workers=2)
        self.assertNotIsInstance(executor, pipeline.SharedPoolExecutor)
        executor.shutdown()


if __name__ == "__main__":
    unittest.main()
