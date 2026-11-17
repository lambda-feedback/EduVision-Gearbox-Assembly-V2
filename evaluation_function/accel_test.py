"""
accel.py daje wyniki IDENTYCZNE z oryginalnym kodem GearBox: losowe pozy
części, maski ROI z zasłonięciami i obszarem poza kadrem, przypadki
brzegowe (bryła za kamerą, rzut poza ROI) i pełne dopasowanie Powella.
"""

import math
import unittest

import numpy as np

from . import accel, pipeline


def _modules():
    pipeline._load_modules()
    import estimate_part_poses
    import fuse_views
    return estimate_part_poses, fuse_views


def _random_axis(rng):
    axis = rng.normal(size=3)
    return axis / np.linalg.norm(axis)


def _scene(epp, models, rng, class_id):
    """Maski ROI jak w build_roi: cel = sylwetka z zaburzonej pozy,
    zasłaniające = losowe prostokąty poza celem, poza kadrem = pas."""
    model = models[class_id]
    K = np.array([[3000.0, 0, 2000], [0, 3000.0, 1500], [0, 0, 1]])
    center = np.array([rng.uniform(-40, 40), rng.uniform(-30, 30),
                       rng.uniform(120, 260)])
    axis = _random_axis(rng)
    with accel.original(epp):
        full = epp.render_silhouette(
            model, epp.T_camera_part(center, axis, model), K, (3000, 4000))
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    target = {"id": "t", "mask_bbox_xyxy": [int(x0), int(y0), int(x1),
                                            int(y1)],
              "mask_offset_px": [int(x0), int(y0)]}
    crops = {"t": full[y0:y1 + 1, x0:x1 + 1].copy()}
    # Część zdjęcia "obcięta" (mniejszy kadr) — obszar poza kadrem w ROI.
    image_size = (4000, 3000) if rng.random() < 0.5 else (
        int(x1) + 5, int(y1) + 5)
    others = []
    for i in range(rng.integers(0, 3)):
        ox = max(0, int(x0 + rng.integers(-150, max(1, x1 - x0))))
        oy = max(0, int(y0 + rng.integers(-150, max(1, y1 - y0))))
        w = int(min(rng.integers(20, 200), image_size[0] - ox))
        h = int(min(rng.integers(20, 200), image_size[1] - oy))
        if w < 1 or h < 1:
            continue
        crop = np.ones((h, w), bool)
        key = f"o{i}"
        crops[key] = crop
        others.append({"id": key, "mask_bbox_xyxy": [ox, oy, ox + w - 1,
                                                     oy + h - 1],
                       "mask_offset_px": [ox, oy]})

    class Cache:
        def get(self, detection):
            return crops[detection["id"]], detection["mask_offset_px"]

    return model, center, axis, K, target, others, Cache(), image_size


class TestAccelIdentical(unittest.TestCase):

    def setUp(self):
        self.epp, self.fuse = _modules()
        if not accel._ORIGINAL:
            self.skipTest("EDUVISION_ACCEL=0")
        with accel.original(self.epp):
            self.models = self.epp.load_models()

    def test_axis_frame(self):
        rng = np.random.default_rng(4)
        for _ in range(3000):
            axis = rng.normal(size=3) * rng.choice([1e-3, 1, 50])
            if rng.random() < 0.3:
                # Wokół progu wyboru wektora pomocniczego (|x| = 0.9).
                axis = np.array([rng.choice([-1, 1]) * rng.uniform(0.89, 0.91),
                                 *rng.normal(size=2) * 0.3])
            with accel.original(self.epp):
                expected = self.epp.axis_frame(axis)
            got = accel.axis_frame(axis)
            self.assertTrue(np.array_equal(expected, got))
            self.assertEqual(expected.dtype, got.dtype)

    def test_transform(self):
        rng = np.random.default_rng(0)
        model = self.models[2]
        for _ in range(3000):
            center = rng.normal(size=3) * 100
            axis = rng.normal(size=3) * rng.choice([1e-3, 1, 50])
            with accel.original(self.epp):
                expected = self.epp.T_camera_part(center, axis, model)
            got = accel.T_camera_part(center, axis, model)
            self.assertTrue(np.array_equal(expected, got))

    def test_render_and_cost(self):
        rng = np.random.default_rng(1)
        checked = 0
        for trial in range(400):
            class_id = list(self.models)[trial % len(self.models)]
            scene = _scene(self.epp, self.models, rng, class_id)
            if scene is None:
                continue
            model, center, axis, K, target, others, cache, size = scene
            with accel.original(self.epp):
                masks, K_roi, _ = self.epp.build_roi(
                    [target], others, size, K, cache)
            fast_masks, fast_K, _ = accel.build_roi([target], others, size,
                                                    K, cache)
            self.assertTrue(np.array_equal(K_roi, fast_K))
            self.assertIsNotNone(fast_masks["_label"])
            shape = masks["target"].shape
            for _ in range(15):
                # Pozy wokół prawdziwej: dobre, złe, za kamerą, poza ROI.
                c = center * rng.uniform(0.7, 1.3) + rng.normal(size=3) * 8
                if rng.random() < 0.05:
                    c[2] = rng.uniform(-50, 1.0)
                n = _random_axis(rng) if rng.random() < 0.3 else axis
                with accel.original(self.epp):
                    T = self.epp.T_camera_part(c, n, model)
                    rendered = self.epp.render_silhouette(model, T, K_roi,
                                                          shape)
                    expected = self.epp.silhouette_cost(rendered, masks)
                got_mask = accel.render_silhouette(model, T, K_roi, shape)
                self.assertTrue(np.array_equal(rendered, got_mask))
                box = accel._LAST["box"]
                if box is not None:
                    x0, y0, x1, y1 = box
                    outside = got_mask.copy()
                    outside[y0:y1, x0:x1] = False
                    self.assertFalse(outside.any())
                got = accel.silhouette_cost(got_mask, fast_masks)
                self.assertEqual(expected, got)
                self.assertEqual([type(v) for v in expected],
                                 [type(v) for v in got])
                checked += 1
        self.assertGreater(checked, 3000)

    def test_mask_from_tensor(self):
        import detect_parts
        import torch
        if "from_tensor" not in accel._ORIGINAL:
            self.skipTest("EDUVISION_ACCEL=0")
        rng = np.random.default_rng(3)
        original = accel._ORIGINAL["from_tensor"]
        cls = detect_parts.MaskCrop
        for trial in range(60):
            mask = np.zeros((300, 400), np.uint8)
            if trial % 10:
                x0, y0 = rng.integers(0, 350), rng.integers(0, 250)
                w, h = rng.integers(1, 60, size=2)
                mask[y0:y0 + h, x0:x0 + w] = rng.integers(
                    0, 2 if trial % 3 else 256, (min(h, 300 - y0),
                                                 min(w, 400 - x0)))
            tensors = [torch.from_numpy(mask)]
            tensors.append(torch.from_numpy(mask.astype(np.float32) / 2))
            for tensor in tensors:
                expected = original(cls, tensor, mask.shape)
                got = cls.from_tensor(tensor, mask.shape)
                if expected is None:
                    self.assertIsNone(got)
                    continue
                self.assertTrue(np.array_equal(expected.crop, got.crop))
                self.assertEqual((expected.x0, expected.y0, expected.area),
                                 (got.x0, got.y0, got.area))

    def test_full_fit_identical(self):
        """fit_candidate (Powell, do 400 ocen) — ten sam wynik."""
        rng = np.random.default_rng(2)
        for trial in range(12):
            class_id = list(self.models)[trial % len(self.models)]
            scene = _scene(self.epp, self.models, rng, class_id)
            if scene is None:
                continue
            model, center, axis, K, target, others, cache, size = scene
            start_c = center * 1.1 + rng.normal(size=3) * 3
            start_n = axis + rng.normal(size=3) * 0.2
            start_n /= np.linalg.norm(start_n)
            with accel.original(self.epp):
                masks, K_roi, _ = self.epp.build_roi([target], others, size,
                                                     K, cache)
                expected = self.epp.fit_candidate(model, start_c, start_n,
                                                  K_roi, masks)
            masks, K_roi, _ = self.epp.build_roi([target], others, size, K,
                                                 cache)
            got = self.epp.fit_candidate(model, start_c, start_n, K_roi,
                                         masks)
            for key in expected:
                if isinstance(expected[key], np.ndarray):
                    self.assertTrue(np.array_equal(expected[key], got[key]))
                else:
                    self.assertEqual(expected[key], got[key], key)
            self.assertTrue(math.isfinite(got["cost"]))


if __name__ == "__main__":
    unittest.main()
