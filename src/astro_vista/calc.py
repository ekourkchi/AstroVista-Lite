"""Arithmetic between buffers: ``[3] = [1] - [2]``.

  [3] = [1] - [2]                 pixel by pixel
  [4] = ([1] - 3246) * 2.5        numbers and parentheses; ^ or ** for powers
  [5] = median([1], [2], [3])     several buffers: pixel by pixel, NaN left out
  [6] = [1] - median([1])         one buffer: its value over the whole image
  [7] = div([1], [2], 0)          a / b, and 0 where b is 0
  [7] = [1] / [2]                 NaN where [2] is 0 (never infinite)
  [8] = [1] * mymask              a mask layer's name: 1 where good, NaN where masked
  [9] = where([1] > 1000, 1, 0)   comparisons give 1 or 0

Buffers are written [n] or Bn.  Variables (set) can be used by name.  Functions:

  mean avg median sum min max std   of several buffers pixel by pixel, of one its value
  div(a, b, fill)
  abs sqrt exp log (natural) log10 sin cos tan arcsin arccos arctan sinh cosh tanh
  floor ceil round minimum maximum where isnan isfinite clip nan_to_num
  good(m)    1 where the mask layer m is good, 0 where it masks
  masked(m)  1 where it masks, 0 where it is good
  nan pi e

Special values are handled for you: dividing by 0, log(0) or an overflow give NaN (never
infinity); + - * / with a NaN pixel give NaN; mean, median, sum, min, max and std leave
NaN out (and give NaN only where every value is NaN).  Masks: 0 is bad, anything else
good.  Every buffer in an expression must have the same size; the result takes the first
buffer's header (and WCS).  The expression is read, not run as Python: only this list
can be used.
"""

from __future__ import annotations

import ast
import re
import warnings
from typing import Callable, Dict, List

import numpy as np

from .cmdbase import CommandError

FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "isnan": np.isnan,
    "isfinite": np.isfinite,
    "clip": np.clip,
    "nan_to_num": np.nan_to_num,
}


def _nansum(values, axis=None):
    """A sum that leaves NaN out, and is NaN (not 0) where every value is NaN."""
    total = np.nansum(values, axis=axis)
    empty = np.all(np.isnan(values), axis=axis)
    return np.where(empty, np.nan, total) if np.ndim(total) else (np.nan if empty else total)


def safe_divide(a, b):
    """a / b, NaN where b is 0 (never infinite)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(all="ignore"):
        return np.where(b == 0, np.nan, a / np.where(b == 0, 1.0, b))


# of several buffers: pixel by pixel; of one: over the whole image (NaN left out)
REDUCTIONS = {
    "mean": np.nanmean,
    "avg": np.nanmean,
    "median": np.nanmedian,
    "sum": _nansum,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
}
CONSTANTS = {"nan": np.nan, "pi": np.pi, "e": np.e}
OPERATORS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: safe_divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}
COMPARISONS = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


def normalize(expression: str) -> str:
    """Bn -> [n] and ^ -> **, so that Python's grammar can read the expression."""
    expression = re.sub(r"\bB(\d+)\b", r"[\1]", expression)
    return expression.replace("^", "**")


class Calculator:
    """Evaluates an expression with ``buffer(n)`` (an Image), ``mask(name)`` (the masked
    pixels of a mask layer, or None) and ``variables``."""

    def __init__(self, buffer: Callable, mask: Callable, variables: Dict[str, float]):
        self.buffer, self.mask, self.variables = buffer, mask, variables
        self.used: List = []  # the buffers met, in order

    def run(self, expression: str):
        try:
            tree = ast.parse(normalize(expression).strip(), mode="eval")
        except SyntaxError:
            raise CommandError(f"cannot read the expression {expression!r}")
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels give NaN
            value = self._value(tree.body)
        # what cannot be a number (x/0, log(0), an overflow) is NaN, never infinite
        if np.ndim(value):
            value = np.where(np.isinf(value), np.nan, value)
        elif np.isinf(value):
            value = float("nan")
        return value

    def _shape_check(self, image) -> None:
        if self.used and image.shape != self.used[0].shape:
            raise CommandError("the buffers in the expression differ in size")
        self.used.append(image)

    def _value(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.List) and len(node.elts) == 1:  # [n]: a buffer
            n = node.elts[0]
            if isinstance(n, ast.Constant) and isinstance(n.value, int):
                image = self.buffer(n.value)
                self._shape_check(image)
                return image.data.astype(np.float64)
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.variables:
                return float(self.variables[name])
            if name in CONSTANTS:
                return CONSTANTS[name]
            masked = self.mask(name)
            if masked is not None:  # a mask layer: 1 where good, NaN where masked
                return np.where(masked, np.nan, 1.0)
            raise CommandError(f"{name!r} is not a buffer, a variable or a mask layer")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self._value(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](self._value(node.left), self._value(node.right))
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in COMPARISONS
        ):
            result = COMPARISONS[type(node.ops[0])](
                self._value(node.left), self._value(node.comparators[0])
            )
            return np.asarray(result, dtype=np.float64)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            return self._call(node.func.id, node.args)
        raise CommandError(
            "arithmetic allows numbers, [n], variables, mask layers, + - * / ** %, "
            "comparisons and the functions in help calc"
        )

    def _call(self, name: str, args):
        if name in ("good", "masked"):
            if len(args) != 1 or not isinstance(args[0], ast.Name):
                raise CommandError(f"{name}() takes the name of a mask layer")
            masked = self.mask(args[0].id)
            if masked is None:
                raise CommandError(f"there is no mask layer {args[0].id!r}")
            return (masked if name == "masked" else ~masked).astype(np.float64)
        if name == "div":
            if len(args) not in (2, 3):
                raise CommandError("div(a, b) or div(a, b, fill)")
            a, b = self._value(args[0]), self._value(args[1])
            fill = self._value(args[2]) if len(args) == 3 else np.nan
            return np.where(np.asarray(b) == 0, fill, safe_divide(a, b))
        if name in REDUCTIONS:
            values = [self._value(a) for a in args]
            if not values:
                raise CommandError(f"{name}() needs at least one value")
            if len(values) == 1:  # one buffer: its value over the image
                return float(REDUCTIONS[name](values[0]))
            stack = np.stack(np.broadcast_arrays(*values))
            return REDUCTIONS[name](stack, axis=0)
        if name in FUNCTIONS:
            return FUNCTIONS[name](*[self._value(a) for a in args])
        raise CommandError(f"unknown function {name!r} (help calc lists them)")
