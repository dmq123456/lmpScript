#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output channels of the dynamic structure factor.

Every channel this program can produce is a real linear form on the 3x3
structure-factor tensor S^{ab}(q,w):

    channel(q,w) = sum_{ab} W_ab * Re S^{ab}(q,w)

so a channel is fully specified by a single real 3x3 weight matrix W. This
module turns the --component command-line tokens into those matrices.

Token grammar
-------------
    token := term ('+' term)*

Separate tokens are separate outputs; terms joined by '+' are summed into one
output. So `--component 1+2 3` yields two curves, S^xx+S^xy and S^xz.

    term        W
    ----        -
    1 .. 9      e_a e_b^T   (row-major: 1=xx, 2=xy, 3=xz, 4=yx, ...)
    xx .. zz    e_a e_b^T
    x, y, z     shorthand for xx, yy, zz
    L           qhat qhat^T          (longitudinal to q)
    T           I - qhat qhat^T      (transverse to q)

L and T depend on the direction of q, so W is built per q-point; every other
term is a constant matrix.

Clipping
--------
S^{ab}(q,w) is a Hermitian positive-semidefinite spectral-density matrix at
each frequency. Therefore sum_ab W_ab S^{ab} >= 0 is guaranteed exactly when W
is symmetric positive-semidefinite -- and that is the condition under which
this module clips negative values (numerical noise) to zero. Channels whose W
is not symmetric PSD (any off-diagonal element, or a mixed group such as 1+2)
are signed quantities and keep their sign.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


_AXIS = {"x": 0, "y": 1, "z": 2}

# Row-major index 1..9 -> (a, b)
_INDEX_AB: dict[str, Tuple[int, int]] = {
    str(1 + 3 * a + b): (a, b) for a in range(3) for b in range(3)
}

_PSD_TOL = 1.0e-12


def _term_weight_constant(term: str) -> np.ndarray | None:
    """Weight matrix of a q-independent term, or None if the term needs qhat."""
    low = term.lower()
    if low in ("l", "t"):
        return None
    if term in _INDEX_AB:
        a, b = _INDEX_AB[term]
    elif low in _AXIS:
        a = b = _AXIS[low]
    elif len(low) == 2 and low[0] in _AXIS and low[1] in _AXIS:
        a, b = _AXIS[low[0]], _AXIS[low[1]]
    else:
        raise ValueError(
            f"Invalid component term {term!r}. Use 1..9, xx..zz, x/y/z, L or T; "
            f"join terms with '+' to sum them (e.g. 1+5+9)."
        )
    w = np.zeros((3, 3), dtype=np.float64)
    w[a, b] = 1.0
    return w


class Channel:
    """One requested output curve: a label plus the weight matrix behind it."""

    def __init__(self, label: str, terms: List[str]) -> None:
        self.label = label
        self.terms = terms
        # Constant part accumulated once; L/T counts handled per q.
        self._const = np.zeros((3, 3), dtype=np.float64)
        self._n_long = 0
        self._n_trans = 0
        for term in terms:
            w = _term_weight_constant(term)
            if w is None:
                if term.lower() == "l":
                    self._n_long += 1
                else:
                    self._n_trans += 1
            else:
                self._const += w

    @property
    def needs_qhat(self) -> bool:
        return self._n_long > 0 or self._n_trans > 0

    def weight(self, q_hat: np.ndarray | None) -> np.ndarray:
        """The 3x3 real weight matrix W for this channel at direction q_hat.

        q_hat is None at q -> 0, where the longitudinal direction is undefined:
        there L contributes nothing and T reduces to the full trace, matching
        the natural Gamma-point limit.
        """
        w = self._const.copy()
        if self._n_long or self._n_trans:
            if q_hat is None:
                w += self._n_trans * np.eye(3)
            else:
                proj = np.outer(q_hat, q_hat)
                w += self._n_long * proj
                w += self._n_trans * (np.eye(3) - proj)
        return w

    def clips(self, q_hat: np.ndarray | None) -> bool:
        """True when W is symmetric positive-semidefinite, so the channel is a
        genuine intensity and negative values are numerical noise."""
        w = self.weight(q_hat)
        if not np.allclose(w, w.T, atol=_PSD_TOL):
            return False
        return bool(np.linalg.eigvalsh(w).min() >= -_PSD_TOL)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Channel({self.label!r})"


def parse_channels(tokens: List[str]) -> List[Channel]:
    """Turn --component tokens into Channel objects, preserving order."""
    if not tokens:
        raise ValueError("At least one --component token is required")
    channels: List[Channel] = []
    for raw in tokens:
        label = raw.strip()
        if not label:
            raise ValueError("Empty --component token")
        terms = [t.strip() for t in label.split("+")]
        if any(not t for t in terms):
            raise ValueError(
                f"Malformed --component token {raw!r}: '+' must join two terms"
            )
        # Validate eagerly so bad input fails before the trajectory is read.
        for t in terms:
            _term_weight_constant(t)
        channels.append(Channel(label, terms))
    return channels


def channel_value(
    s_ab: np.ndarray,
    channel: Channel,
    q_hat: np.ndarray | None,
) -> np.ndarray:
    """Contract S^{ab}(q,w) with the channel weight -> real spectrum (nfreq,)."""
    w = channel.weight(q_hat)
    val = np.real(np.einsum("wab,ab->w", s_ab, w))
    if channel.clips(q_hat):
        np.maximum(val, 0.0, out=val)
    return val


__all__ = ["Channel", "parse_channels", "channel_value"]
