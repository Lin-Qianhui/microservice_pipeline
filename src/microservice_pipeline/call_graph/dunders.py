"""Operator-to-dunder tables.

Python syntax can invoke methods implicitly. These tables translate AST
operator node types into the corresponding data-model ("dunder") methods so,
for example, ``left + right`` can produce an edge to ``left.__add__``.

Pure data with no dependencies beyond ``ast`` -- the collector imports these to
decide which method an operator node stands for.
"""


from __future__ import annotations

import ast

BINOP_DUNDER_METHODS = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.MatMult: "__matmul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitOr: "__or__",
    ast.BitXor: "__xor__",
    ast.BitAnd: "__and__",
}

REVERSE_BINOP_DUNDER_METHODS = {
    ast.Add: "__radd__",
    ast.Sub: "__rsub__",
    ast.Mult: "__rmul__",
    ast.MatMult: "__rmatmul__",
    ast.Div: "__rtruediv__",
    ast.FloorDiv: "__rfloordiv__",
    ast.Mod: "__rmod__",
    ast.Pow: "__rpow__",
    ast.LShift: "__rlshift__",
    ast.RShift: "__rrshift__",
    ast.BitOr: "__ror__",
    ast.BitXor: "__rxor__",
    ast.BitAnd: "__rand__",
}

AUGASSIGN_DUNDER_METHODS = {
    ast.Add: "__iadd__",
    ast.Sub: "__isub__",
    ast.Mult: "__imul__",
    ast.MatMult: "__imatmul__",
    ast.Div: "__itruediv__",
    ast.FloorDiv: "__ifloordiv__",
    ast.Mod: "__imod__",
    ast.Pow: "__ipow__",
    ast.LShift: "__ilshift__",
    ast.RShift: "__irshift__",
    ast.BitOr: "__ior__",
    ast.BitXor: "__ixor__",
    ast.BitAnd: "__iand__",
}

COMPARE_DUNDER_METHODS = {
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
}

UNARY_DUNDER_METHODS = {
    ast.UAdd: "__pos__",
    ast.USub: "__neg__",
    ast.Invert: "__invert__",
}