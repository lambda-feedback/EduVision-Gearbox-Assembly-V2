"""
Pula procesów roboczych (os.fork + gniazda socketpair) BEZ semaforów
POSIX. multiprocessing.Queue i ProcessPoolExecutor potrzebują /dev/shm,
którego nie ma na AWS Lambda — tam etapy "cad" i "fuse" liczyły się
w jednym procesie. Ta pula działa wszędzie, gdzie jest fork() (Linux).

Procesy robocze są kopiami procesu, który tworzy pulę (bez kosztu importu
bibliotek; pamięć współdzielona copy-on-write). Pulę trzeba utworzyć,
ZANIM proces uruchomi wątki obliczeniowe (torch/OpenMP, OpenCV) — kopia
procesu z działającą pulą wątków potrafi się zawiesić (worker.py tworzy
ją na samym początku zadania).

Interfejs jak concurrent.futures: submit() -> Future, map(), shutdown().
Wyjątek w zadaniu trafia do Future (z tekstem stosu wywołań procesu
roboczego jako __cause__); śmierć procesu roboczego (np. brak pamięci)
kończy wszystkie zadania wyjątkiem BrokenProcessPool — jak w
ProcessPoolExecutor.
"""

import os
import pickle
import signal
import threading
import traceback
from collections import deque
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.connection import Pipe, wait

_PROTOCOL = pickle.HIGHEST_PROTOCOL


class RemoteTraceback(Exception):
    """Stos wywołań z procesu roboczego (jako __cause__ wyjątku)."""

    def __init__(self, text):
        super().__init__(text)
        self.text = text

    def __str__(self):
        return self.text


def _reply(ok, value, tb_text=""):
    try:
        return pickle.dumps((ok, value, tb_text), protocol=_PROTOCOL)
    except Exception as exc:        # wynik / wyjątek nie do przesłania
        error = RuntimeError(f"cannot send the result of a pool task: "
                             f"{exc!r}; original: {value!r}"[:2000])
        return pickle.dumps((False, error, tb_text), protocol=_PROTOCOL)


def _serve(conn):
    """Pętla procesu roboczego: zadanie -> wynik, do końca połączenia."""
    while True:
        try:
            data = conn.recv_bytes()
        except (EOFError, OSError):
            return
        try:
            fn, args, kwargs = pickle.loads(data)
            payload = _reply(True, fn(*args, **kwargs))
        except BaseException as exc:    # także SystemExit z sys.exit()
            payload = _reply(False, exc, traceback.format_exc())
        conn.send_bytes(payload)


class ForkPool:
    """Pula `workers` procesów. on_start() — wywoływane w każdym procesie
    roboczym na starcie (np. zakończenie razem z rodzicem)."""

    def __init__(self, workers, on_start=None):
        if not hasattr(os, "fork"):
            raise OSError("fork() is not available")
        self._lock = threading.Lock()
        self._pending = deque()         # (Future, dane zadania)
        self._busy = {}                 # połączenie -> Future
        self._broken = None
        self._closing = False
        self._workers = []              # (pid, połączenie)
        for _ in range(max(1, int(workers))):
            parent_end, child_end = Pipe(duplex=True)
            pid = os.fork()
            if pid == 0:                # proces roboczy
                code = 0
                try:
                    parent_end.close()
                    # Końcówki wcześniejszych procesów (EOF musi do nich
                    # dotrzeć, gdy rodzic zamknie swoje).
                    for _, conn in self._workers:
                        conn.close()
                    if on_start is not None:
                        on_start()
                    _serve(child_end)
                except BaseException:
                    code = 1
                finally:
                    os._exit(code)
            child_end.close()
            self._workers.append((pid, parent_end))
        self._idle = [conn for _, conn in self._workers]
        self._wake_r, self._wake_w = os.pipe()
        # Wątek dyspozytora — po utworzeniu procesów (nie jest kopiowany).
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="forkpool")
        self._thread.start()

    @property
    def size(self):
        return len(self._workers)

    @property
    def pids(self):
        return [pid for pid, _ in self._workers]

    # ------------------------------------------------------------ zadania

    def submit(self, fn, *args, **kwargs):
        payload = pickle.dumps((fn, args, kwargs), protocol=_PROTOCOL)
        future = Future()
        with self._lock:
            if self._broken is not None:
                raise BrokenProcessPool(self._broken)
            if self._closing:
                raise RuntimeError("cannot submit after shutdown")
            self._pending.append((future, payload))
        self._wake()
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        return [future.result(timeout) for future in futures]

    def shutdown(self, wait=True, cancel_futures=False):
        with self._lock:
            if cancel_futures:
                for future, _ in self._pending:
                    future.cancel()
                self._pending.clear()
            self._closing = True
        if not wait:
            self._kill()                # dyspozytor dostanie EOF i skończy
        self._wake()
        self._thread.join()
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False

    # -------------------------------------------------------- dyspozytor

    def _wake(self):
        try:
            os.write(self._wake_w, b"x")
        except OSError:
            pass

    def _dispatch(self):
        """Zadania oczekujące -> wolne procesy (pod blokadą)."""
        while self._pending and self._idle:
            future, payload = self._pending.popleft()
            if not future.set_running_or_notify_cancel():
                continue
            conn = self._idle.pop()
            try:
                conn.send_bytes(payload)
            except OSError as exc:
                self._busy[conn] = future
                return f"a pool process died ({exc})"
            self._busy[conn] = future
        return None

    def _loop(self):
        while True:
            with self._lock:
                error = self._dispatch()
                if error is None and self._closing and not self._busy \
                        and not self._pending:
                    return
                busy = list(self._busy)
            if error is not None:
                self._fail(error)
                return
            for item in wait(busy + [self._wake_r]):
                if item == self._wake_r:
                    os.read(self._wake_r, 4096)
                    continue
                try:
                    ok, value, tb_text = pickle.loads(item.recv_bytes())
                except (EOFError, OSError):
                    self._fail("a pool process terminated abruptly "
                               "(killed or out of memory?)")
                    return
                except Exception as exc:
                    ok, value, tb_text = False, exc, traceback.format_exc()
                with self._lock:
                    future = self._busy.pop(item)
                    self._idle.append(item)
                if ok:
                    future.set_result(value)
                else:
                    if tb_text and isinstance(value, BaseException):
                        value.__cause__ = RemoteTraceback(tb_text)
                    future.set_exception(value)

    def _fail(self, message):
        """Proces roboczy padł: wszystkie zadania kończą się wyjątkiem."""
        with self._lock:
            self._broken = message
            futures = list(self._busy.values()) + [
                f for f, _ in self._pending]
            self._busy.clear()
            self._pending.clear()
        self._kill()
        for future in futures:
            if not future.done():
                future.set_exception(BrokenProcessPool(message))

    # ----------------------------------------------------------- koniec

    def _kill(self):
        for pid, _ in self._workers:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def _close(self):
        for _, conn in self._workers:
            try:
                conn.close()            # EOF -> proces roboczy kończy się
            except OSError:
                pass
        for pid, _ in self._workers:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        for fd in (self._wake_r, self._wake_w):
            try:
                os.close(fd)
            except OSError:
                pass
        self._workers = []
        self._idle = []
