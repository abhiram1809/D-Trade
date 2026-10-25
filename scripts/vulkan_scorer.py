"""Native Vulkan forward, softmax backward, gradient reduction and Adam updates."""
import ctypes as C
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class VulkanScorer:
    def __init__(self, width, choices=5, batch_size=256):
        self.width, self.choices, self.batch_size = width, choices, batch_size
        self.lib = C.CDLL(str(ROOT / 'native/libscorer.so'))
        self.fp, self.ip = C.POINTER(C.c_float), C.POINTER(C.c_uint32)
        self.lib.dt_create.argtypes = [C.c_char_p, C.c_uint32, C.c_uint32, C.c_uint32]
        self.lib.dt_create.restype = C.c_void_p
        self.lib.dt_set.argtypes = [C.c_void_p, self.fp]
        self.lib.dt_get.argtypes = [C.c_void_p, self.fp]
        self.lib.dt_run.argtypes = [C.c_void_p, self.fp, self.ip, self.fp, self.fp, C.c_uint32, C.c_float, C.c_float, C.c_int]
        self.lib.dt_name.argtypes = [C.c_void_p]
        self.lib.dt_name.restype = C.c_char_p
        self.lib.dt_destroy.argtypes = [C.c_void_p]
        self.handle = self.lib.dt_create(str(ROOT/'native/scorer.spv').encode(), width, choices, batch_size)
        if not self.handle:
            raise RuntimeError('Vulkan scorer initialization failed; no CPU fallback')
        self.device_name = self.lib.dt_name(self.handle).decode()

    def set_weights(self, w):
        w = np.ascontiguousarray(w, dtype='float32').reshape(self.width)
        self.lib.dt_set(self.handle, w.ctypes.data_as(self.fp))

    def get_weights(self):
        w = np.empty(self.width, dtype='float32')
        self.lib.dt_get(self.handle, w.ctypes.data_as(self.fp))
        return w

    def run(self, x, y=None, lr=0., l2=0., update=False):
        x = np.ascontiguousarray(x, dtype='float32')
        if x.ndim != 3 or x.shape[1:] != (self.choices, self.width):
            raise ValueError(f'Expected [batch,{self.choices},{self.width}], got {x.shape}')
        if update and y is None:
            raise ValueError('Updates require labels')
        labels = None if y is None else np.ascontiguousarray(y, dtype='uint32')
        if labels is not None and (labels.shape != (len(x),) or np.any(labels >= self.choices)):
            raise ValueError('Invalid class labels')
        z = np.empty((len(x), self.choices), dtype='float32')
        g = np.empty(self.width, dtype='float32') if y is not None else None
        status = self.lib.dt_run(self.handle, x.ctypes.data_as(self.fp), labels.ctypes.data_as(self.ip) if labels is not None else None, z.ctypes.data_as(self.fp), g.ctypes.data_as(self.fp) if g is not None else None, len(x), lr, l2, int(update))
        if status:
            raise RuntimeError('Vulkan dispatch failed')
        return z, g

    def close(self):
        if self.handle:
            self.lib.dt_destroy(self.handle)
            self.handle = None


def parity_check():
    import hashlib
    import json
    import time
    from scipy.special import softmax
    rng = np.random.default_rng(42)
    reports = []
    for n, d in [(19, 37), (257, 1024), (129, 5120)]:
        x = rng.normal(0, .2, (n, 5, d)).astype('float32')
        w = rng.normal(0, .05, d).astype('float32')
        y = rng.integers(0, 5, n, dtype='uint32')
        engine = VulkanScorer(d, batch_size=512)
        engine.set_weights(w)
        z_cpu = x @ w
        error = softmax(z_cpu, axis=1)
        error[np.arange(n), y] -= 1
        g_cpu = np.einsum('nk,nkd->d', error, x) / n
        z, g = engine.run(x, y)
        lr, l2 = .001, .01
        reg = g_cpu + l2 * w
        expected_w = w - lr * reg / (np.sqrt(reg * reg) + 1e-8)
        engine.run(x, y, lr, l2, update=True)
        got_w = engine.get_weights()
        # First step Adam correction can amplify tiny errors for near-zero gradients.
        np.testing.assert_allclose(z, z_cpu, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(g, g_cpu, atol=2e-6, rtol=2e-3)
        np.testing.assert_allclose(got_w, expected_w, atol=3e-5, rtol=3e-3)
        report = {'device': engine.device_name, 'batch': n, 'width': d, 'forward_max_abs': float(np.max(abs(z-z_cpu))), 'gradient_max_abs': float(np.max(abs(g-g_cpu))), 'adam_update_max_abs': float(np.max(abs(got_w-expected_w))), 'weight_delta_l2': float(np.linalg.norm(got_w-w)), 'passed': True}
        reports.append(report)
        engine.close()
    out = {'checked_at_unix': time.time(), 'shader_sha256': hashlib.sha256((ROOT/'native/scorer.comp').read_bytes()).hexdigest(), 'tests': reports}
    (ROOT/'results/vulkan_parity.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    parity_check()
