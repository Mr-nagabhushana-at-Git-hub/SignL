# src/majorSignL/utils/one_euro_filter.py

import math
import numpy as np

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        def alpha(cutoff):
            te = 1.0 / self.freq
            tau = 1.0 / (2 * math.pi * cutoff)
            return 1.0 / (1.0 + tau / te)

        dx = (x - self.x_prev) * self.freq
        if self.dx_prev is not None:
            alpha_d = alpha(self.d_cutoff)
            dx = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        self.dx_prev = dx

        cutoff = self.min_cutoff + self.beta * np.abs(dx)
        alpha_val = alpha(cutoff)
        x_filtered = alpha_val * x + (1 - alpha_val) * self.x_prev
        self.x_prev = x_filtered
        return x_filtered