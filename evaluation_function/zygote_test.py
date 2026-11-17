"""
Tryb fork (Linux): proces-matka, kopie dla zgłoszeń, pula procesów zadania,
limit czasu, awaria kopii (powtórka w procesie jednorazowym) — i wyniki
identyczne z procesem jednorazowym. Zdjęcia syntetyczne (AprilTag
z cv2.aruco), więc bez danych zewnętrznych.
"""

import base64
import os
import re
import signal
import sys
import threading
import unittest
from unittest import mock

import cv2
import numpy as np

from . import evaluation


def _photo(seed):
    """Zdjęcie 2000x1500 z AprilTagiem tag36h11 id 0 (perspektywa)."""
    rng = np.random.default_rng(seed)
    tag = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        0, 400)
    tag = cv2.cvtColor(cv2.copyMakeBorder(tag, 50, 50, 50, 50,
                                          cv2.BORDER_CONSTANT, value=255),
                       cv2.COLOR_GRAY2BGR)
    height, width = 1500, 2000
    image = cv2.GaussianBlur(cv2.resize(
        rng.integers(60, 200, (30, 40, 3), dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_CUBIC), (0, 0), 3)
    size = tag.shape[0]
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    skew = rng.uniform(0.6, 0.9)
    dst = (np.float32([[-1, -skew], [1, -skew], [1.15, skew], [-1.15, skew]])
           * 180 + [1000 + rng.integers(-200, 200), 750]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tag, M, (width, height))
    inside = cv2.warpPerspective(np.full(tag.shape[:2], 255, np.uint8), M,
                                 (width, height)) > 0
    image[inside] = warped[inside]
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"data": base64.b64encode(data.tobytes()).decode(),
            "name": f"IMG_{seed}.jpg"}


def _strip_debug(feedback):
    """Feedback bez sekcji Debug i bez logu analizy (ścieżki, czasy)."""
    return re.sub(r"```.*?```", "", feedback.split("### Debug")[0],
                  flags=re.S)


def _children(pid):
    """Procesy potomne pid: [(pid, stan, cmdline)] — tylko do komunikatu
    przy błędzie testu."""
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as file:
                # pid (comm) state ppid ... — comm może zawierać spacje.
                state, ppid = file.read().rsplit(")", 1)[1].split()[:2]
            if int(ppid) != pid:
                continue
            with open(f"/proc/{name}/cmdline", "rb") as file:
                cmdline = file.read().replace(b"\0", b" ").decode(
                    errors="replace").strip()
        except (OSError, IndexError, ValueError):
            continue
        found.append((int(name), state, cmdline[:120]))
    return sorted(found)


@unittest.skipUnless(sys.platform.startswith("linux"), "fork: Linux only")
class TestZygote(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.response = [_photo(seed) for seed in range(3)]
        cls.env = mock.patch.dict(os.environ, {
            "EDUVISION_WORKER_MODE": "fork", "EDUVISION_WORKERS": "2",
            "YOLO_OFFLINE": "1"})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        with evaluation._ZYGOTE_LOCK:
            if evaluation._ZYGOTE is not None:
                evaluation._ZYGOTE.kill()
                evaluation._ZYGOTE = None
        evaluation._ZYGOTE_FAILURES = 0
        cls.env.stop()

    def _evaluate(self, **env):
        with mock.patch.dict(os.environ, env):
            return evaluation.evaluation_function(
                self.response, "etap_pelny",
                {"image_mode": "none", "debug": True}).to_dict()

    def _crash_job_child(self, sig, sent):
        """Patch: 0,5 s po starcie zadania wysyła sygnał jego kopii. PID
        z protokołu (@@STARTED <pid>), jak przy limicie czasu w serwerze —
        nie zgadujemy go z listy procesów. Do sent trafia (pid, procesy
        potomne procesu-matki w tej chwili) — do komunikatu przy błędzie."""
        original = evaluation._Zygote._next

        def kill(pid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

        def _next(zygote, deadline):
            line = original(zygote, deadline)
            if line and line.startswith("@@STARTED ") and not sent:
                pid = int(line.split()[1])
                sent.append((pid, _children(zygote.proc.pid)))
                threading.Timer(0.5, kill, (pid,)).start()
            return line
        return mock.patch.object(evaluation._Zygote, "_next", _next)

    def test_fork_matches_once_survives_timeout_and_crash(self):
        once = self._evaluate(EDUVISION_WORKER_MODE="once")
        self.assertIn("worker_mode: `once`", once["feedback"])

        first = self._evaluate()
        self.assertIn("worker_mode: `fork`", first["feedback"])
        self.assertIn("fork pool (2)", first["feedback"])
        second = self._evaluate()
        self.assertIn("worker_mode: `fork`", second["feedback"])
        for result in (first, second):
            self.assertEqual(result["is_correct"], once["is_correct"])
            self.assertEqual(_strip_debug(result["feedback"]),
                             _strip_debug(once["feedback"]))

        # Limit czasu: zadanie zabite, proces-matka dalej działa.
        timed_out = self._evaluate(EDUVISION_TIME_LIMIT_S="1")
        self.assertIn("took too long", timed_out["feedback"])
        self.assertTrue(evaluation._ZYGOTE.alive())

        # Awaria kopii (np. SIGABRT po odmrożeniu na AWS Lambda): to samo
        # zadanie w procesie jednorazowym, ten sam wynik; proces-matka
        # zostaje (jedna awaria nie wyłącza trybu fork).
        sent = []
        with self._crash_job_child(signal.SIGABRT, sent):
            crashed = self._evaluate()
        self.assertTrue(sent, "zadanie nie wystartowało w procesie-matce")
        self.assertIn("zygote_failure_return_code: `-6`", crashed["feedback"],
                      f"SIGABRT do kopii {sent[0][0]}; procesy potomne "
                      f"procesu-matki: {sent[0][1]}")
        self.assertIn("worker_mode: `once`", crashed["feedback"])
        self.assertEqual(_strip_debug(crashed["feedback"]),
                         _strip_debug(once["feedback"]))
        self.assertEqual(evaluation._ZYGOTE_FAILURES, 1)

        after = self._evaluate()
        self.assertIn("worker_mode: `fork`", after["feedback"])
        self.assertEqual(_strip_debug(after["feedback"]),
                         _strip_debug(once["feedback"]))


if __name__ == "__main__":
    unittest.main()
