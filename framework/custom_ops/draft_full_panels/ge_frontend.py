"""Static GE tensor backend for the public operations used by the repository's DraftGraph.forward."""

from contextlib import contextmanager
import hashlib
import math
import numbers
import operator
from pathlib import Path

import numpy as np
import torch

# GE TensorDescriptor.dtype (proto enum) differs from Cast.dst_type / AttrDef.dt (C enum).
PROTO = {
    torch.float32: 1,
    torch.float16: 2,
    torch.int8: 3,
    torch.int16: 5,
    torch.int32: 7,
    torch.int64: 8,
    torch.bool: 11,
}
GE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.int8: 2,
    torch.int32: 3,
    torch.int16: 6,
    torch.int64: 9,
    torch.bool: 12,
}


def desc(d, name, dtype, shape, layout="ND", origin=None):
    d.name = name
    d.dtype = PROTO[dtype]
    d.layout = layout
    d.shape.SetInParent()
    d.shape.dim.extend(shape)
    d.attr["format_for_int"].i = 29 if layout == "FRACTAL_NZ" else 2
    d.attr["origin_format_for_int"].i = 2
    d.attr["origin_shape"].list.val_type = 2
    d.attr["origin_shape"].list.i.extend(shape if origin is None else origin)
    d.attr["origin_format_is_set"].b = True
    d.attr["origin_shape_initialized"].b = True


class Graph:
    def __init__(self, types, name):
        self.model = types[0]()
        self.model.name = name
        self.graph = self.model.graph.add(name=name)
        self.module = "setup"
        self.layer = None
        self.owners = {}
        self.ops = {}
        self.literals = {}
        self.projections = {}

    @contextmanager
    def scope(self, name):
        old = self.module
        self.module = name
        try:
            yield
        finally:
            self.module = old

    def node(self, kind, name=None):
        name = name or ("df_" + self.module.replace(".", "_") + "_" + str(len(self.graph.op)) + "_" + kind)
        if name in self.ops:
            raise ValueError("duplicate Draft graph node " + name)
        op = self.graph.op.add(name=name, type=kind, id=len(self.graph.op))
        self.ops[name] = op
        self.owners[name] = self.module
        return op

    def data(self, name, dtype, shape, index):
        op = self.node("Data", name)
        op.attr["index"].i = index
        desc(op.output_desc.add(), "y", dtype, shape)
        op.input_desc.add().CopyFrom(op.output_desc[0])
        self.graph.input.append(name + ":0")
        return Value(self, name, dtype, shape)

    def const(self, value, dtype=None):
        if isinstance(value, Value):
            return value if dtype is None else value.to(dtype)
        tensor = torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous()
        buf = tensor.numpy().tobytes()
        key = (tensor.dtype, tuple(tensor.shape), hashlib.sha256(buf).digest())
        if key not in self.literals:
            self.literals[key] = self.blob(buf, tensor.dtype, tuple(tensor.shape))
        return self.literals[key]

    def blob(self, buf, dtype, shape, *, name=None, layout="ND", origin=None):
        if len(buf) != math.prod(shape) * torch.empty((), dtype=dtype).element_size():
            raise ValueError("bad Const extent " + str(name))
        op = self.node("Const", name)
        desc(op.output_desc.add(), "y", dtype, shape, layout, origin)
        op.attr["value"].t.desc.CopyFrom(op.output_desc[0])
        op.attr["value"].t.data = buf
        op.attr["_out_shape_locked"].b = True
        return Value(self, op.name, dtype, shape)

    def external(self, path, shape, name):
        path = Path(path).resolve(strict=True)
        if path.stat().st_size != math.prod(shape) * 2:
            raise ValueError("wrong FP16 external weight extent")
        op = self.node("FileConstant", name)
        op.attr["shape"].list.val_type = 2
        op.attr["shape"].list.i.extend(shape)
        op.attr["dtype"].dt = GE[torch.float16]
        op.attr["file_path"].s = str(path).encode()
        desc(op.output_desc.add(), "y", torch.float16, shape)
        return Value(self, name, torch.float16, shape)

    def emit(self, kind, inputs, shape, dtype, attrs=None, *, name=None):
        bound = [(port, self.const(x)) for port, x in inputs]
        op = self.node(kind, name)
        for port, v in bound:
            op.input.append(v.name + ":0")
            d = op.input_desc.add()
            d.CopyFrom(self.ops[v.name].output_desc[0])
            d.name = port
        for key, (typ, val) in (attrs or {}).items():
            attr = op.attr[key]
            if typ == "ints":
                attr.list.val_type = 2
                attr.list.i.extend(val)
            else:
                setattr(attr, typ, val)
        desc(op.output_desc.add(), "y", dtype, shape)
        return Value(self, op.name, dtype, shape)

    def outputs(self, values, names):
        if len(names) != len(values):
            raise ValueError("public output arity")
        op = self.node("NetOutput", "NetOutput")
        for i, (name, v) in enumerate(zip(names, values)):
            op.input.append(v.name + ":0")
            out = op.input_desc.add()
            out.CopyFrom(self.ops[v.name].output_desc[0])
            out.name = name
            self.graph.output.append(v.name + ":0")
        return [
            dict(name=n, dtype=str(v.dtype).split(".")[-1], shape=list(v.shape))
            for n, v in zip(names, values)
        ]


def dimensions(shape, args):
    requested = list(args[0] if len(args) == 1 and isinstance(args[0], (list, tuple, torch.Size)) else args)
    if requested.count(-1) > 1:
        raise ValueError("ambiguous reshape")
    if -1 in requested:
        p = requested.index(-1)
        requested[p] = 1
        prod = math.prod(requested)
        if prod <= 0 or math.prod(shape) % prod:
            raise ValueError("invalid inferred static dimension")
        requested[p] = math.prod(shape) // prod
    if math.prod(requested) != math.prod(shape) or any(v < 0 for v in requested):
        raise ValueError("invalid reshape")
    return tuple(requested)


class Value:
    def __init__(self, g, name, dtype, shape):
        self.g = g
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)

    @property
    def device(self):
        return torch.device("cpu")  # graph construction constants on host

    @property
    def ndim(self):
        return len(self.shape)

    def contiguous(self):
        return self

    def reshape(self, *args):
        shape = dimensions(self.shape, args)
        if shape == self.shape:
            return self
        return self.g.emit(
            "Reshape", [("x", self), ("shape", torch.tensor(shape, dtype=torch.int32))], shape, self.dtype
        )

    def reshape_as(self, other):
        return self.reshape(other.shape)

    view = reshape

    def unsqueeze(self, dim):
        dim = dim % (self.ndim + 1)
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    def to(self, dtype=None, **kwargs):
        if dtype == self.dtype:
            return self
        if dtype not in PROTO:
            raise ValueError("unexpected graph Cast " + str(dtype))
        return self.g.emit("Cast", [("x", self)], self.shape, dtype, {"dst_type": ("i", GE[dtype])})

    def float(self):
        return self.to(torch.float32)

    def long(self):
        return self.to(torch.int64)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        count = sum(v is not None and v is not Ellipsis for v in key)
        if count > self.ndim or sum(v is Ellipsis for v in key) > 1:
            raise IndexError(key)
        if Ellipsis in key:
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - count) + key[i + 1 :]
        else:
            key = key + (slice(None),) * (self.ndim - count)
        offsets = []
        sizes = []
        shape = []
        for v in key:
            if v is None:
                shape.append(1)
                continue
            n = self.shape[len(sizes)]
            if isinstance(v, int):
                j = v + n if v < 0 else v
                if not 0 <= j < n:
                    raise IndexError(v)
                offsets.append(j)
                sizes.append(1)
            else:
                lo, hi, step = v.indices(n)
                if step != 1 or hi <= lo:
                    raise ValueError("unsupported empty/strided graph slice")
                offsets.append(lo)
                sizes.append(hi - lo)
                shape.append(hi - lo)
        out = self
        if tuple(sizes) != self.shape:
            out = self.g.emit(
                "Slice",
                [
                    ("x", self),
                    ("offsets", torch.tensor(offsets, dtype=torch.int32)),
                    ("size", torch.tensor(sizes, dtype=torch.int32)),
                ],
                sizes,
                self.dtype,
            )
        return out.reshape(shape)

    def split(self, size, dim=-1):
        dim %= self.ndim
        return tuple(
            self[(slice(None),) * dim + (slice(i, min(i + size, self.shape[dim])),)]
            for i in range(0, self.shape[dim], size)
        )

    def chunk(self, n, dim=-1):
        return self.split((self.shape[dim] + n - 1) // n, dim)

    def transpose(self, a, b):
        perm = list(range(self.ndim))
        perm[a], perm[b] = perm[b], perm[a]
        return self.g.emit(
            "Transpose",
            [("x", self), ("perm", torch.tensor(perm, dtype=torch.int32))],
            tuple(self.shape[i] for i in perm),
            self.dtype,
        )

    def expand(self, *dims):
        pad = (1,) * (len(dims) - self.ndim) + self.shape
        shape = tuple(d if x == -1 else x for d, x in zip(pad, dims))
        if tuple(torch.broadcast_shapes(self.shape, shape)) != shape:
            raise ValueError("invalid broadcast")
        if shape == self.shape:
            return self
        return self.g.emit(
            "BroadcastTo", [("x", self), ("shape", torch.tensor(shape, dtype=torch.int32))], shape, self.dtype
        )

    def unary(self, op):
        return self.g.emit(op, [("x", self)], self.shape, self.dtype)

    def cos(self):
        return self.unary("Cos")

    def sin(self):
        return self.unary("Sin")

    def pow(self, exponent):
        if exponent != 2:
            raise ValueError("only RMS square is supported")
        return self.unary("Square")

    def mean(self, dim, keepdim=False):
        dim %= self.ndim
        shape = list(self.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        return self.g.emit(
            "ReduceMean",
            [("x", self), ("axes", torch.tensor([dim], dtype=torch.int32))],
            shape,
            self.dtype,
            {"keep_dims": ("b", keepdim)},
        )

    def masked_fill(self, mask, val):
        return self.where(mask, self.g.const(val, self.dtype), self)

    def __neg__(self):
        return self.unary("Neg")

    def __invert__(self):
        return self.unary("LogicalNot")

    def binary(self, op, a, b, boolean=False):
        sample = lambda v: torch.empty(v.shape, device="meta", dtype=v.dtype) if isinstance(v, Value) else v
        dtype = torch.result_type(sample(a), sample(b))
        a, b = self.g.const(a, dtype), self.g.const(b, dtype)
        shape = torch.broadcast_shapes(a.shape, b.shape)
        return self.g.emit(op, [("x1", a), ("x2", b)], shape, torch.bool if boolean else dtype)

    def __add__(self, b):
        return self.binary("Add", self, b)

    def __radd__(self, a):
        return self.binary("Add", a, self)

    def __sub__(self, b):
        return self.binary("Sub", self, b)

    def __rsub__(self, a):
        return self.binary("Sub", a, self)

    def __mul__(self, b):
        return self.binary("Mul", self, b)

    def __rmul__(self, a):
        return self.binary("Mul", a, self)

    def __and__(self, b):
        return self.binary("LogicalAnd", self, b)

    def __lt__(self, b):
        return self.binary("Less", self, b, True)

    def __gt__(self, b):
        return self.binary("Greater", self, b, True)

    def __le__(self, b):
        return self.binary("LessEqual", self, b, True)

    def __ge__(self, b):
        return self.binary("GreaterEqual", self, b, True)

    def where(self, condition, x, y):
        x, y = self.g.const(x), self.g.const(y)
        condition = self.g.const(condition)
        if x.dtype != y.dtype or condition.dtype != torch.bool:
            raise ValueError("Select dtype drift")
        shape = torch.broadcast_shapes(x.shape, y.shape, condition.shape)
        return self.g.emit("SelectV2", [("condition", condition), ("x1", x), ("x2", y)], shape, x.dtype)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kw = kwargs or {}
        name = func.__name__
        if name == "cat":
            values = args[0]
            sample = next(v for v in values if isinstance(v, Value))
            values = [sample.g.const(v) for v in values]
            axis = kw.get("dim", args[1] if len(args) > 1 else 0) % values[0].ndim
            shape = list(values[0].shape)
            shape[axis] = sum(v.shape[axis] for v in values)
            if any(
                v.dtype != values[0].dtype
                or v.shape[:axis] + v.shape[axis + 1 :]
                != values[0].shape[:axis] + values[0].shape[axis + 1 :]
                for v in values
            ):
                raise ValueError("incompatible Concat operands")
            return sample.g.emit(
                "ConcatD",
                [(f"x{i}", v) for i, v in enumerate(values)],
                shape,
                values[0].dtype,
                {"N": ("i", len(values)), "concat_dim": ("i", axis)},
            )
        v = next(v for v in args if isinstance(v, Value))
        if name in ("lt", "le", "gt", "ge", "add", "mul", "sub"):
            kind = {
                "lt": "Less",
                "le": "LessEqual",
                "gt": "Greater",
                "ge": "GreaterEqual",
                "add": "Add",
                "mul": "Mul",
                "sub": "Sub",
            }[name]
            return v.binary(kind, *args[:2], name in ("lt", "le", "gt", "ge"))
        if name == "zeros_like":
            return v.g.const(0, v.dtype).expand(*v.shape)
        if name == "where":
            return v.where(*args)
        if name in ("cos", "sin", "rsqrt", "silu"):
            return v.unary({"cos": "Cos", "sin": "Sin", "rsqrt": "Rsqrt", "silu": "Swish"}[name])
        if name == "softmax":
            x = args[0].to(kw.get("dtype", v.dtype))
            axis = kw.get("dim", args[1] if len(args) > 1 else -1)
            return v.g.emit("SoftmaxV2", [("x", x)], x.shape, x.dtype, {"axes": ("ints", [axis])})
        if name == "matmul":
            a, b = args
            if a.ndim != 4 or b.ndim != 4 or a.shape[:2] != b.shape[:2] or a.shape[-1] != b.shape[-2]:
                raise ValueError("unexpected attention matrix contract")
            return v.g.emit(
                "BatchMatMulV2",
                [("x1", a), ("x2", b)],
                a.shape[:-1] + (b.shape[-1],),
                a.dtype,
                {"adj_x1": ("b", False), "adj_x2": ("b", False)},
            )
        if name == "linear":
            x, w = args[:2]
            flat = x.reshape(-1, x.shape[-1])
            if w.ndim != 2 or x.shape[-1] != w.shape[1]:
                raise ValueError("unexpected Head matrix")
            y = v.g.emit(
                "MatMul",
                [("x1", flat), ("x2", w)],
                (flat.shape[0], w.shape[0]),
                x.dtype,
                {"transpose_x1": ("b", False), "transpose_x2": ("b", True)},
            )
            return y.reshape(x.shape[:-1] + (w.shape[0],))
        if name == "argmax":
            dim = kw.get("dim", args[1] if len(args) > 1 else -1) % v.ndim
            # Keep the public PyTorch INT64 Top1. Use the conventional GE INT32 device ArgMax first.
            x = v.g.emit(
                "ArgMaxV2",
                [("x", v), ("dimension", torch.tensor(dim, dtype=torch.int32))],
                v.shape[:dim] + v.shape[dim + 1 :],
                torch.int32,
                {"dtype": ("dt", GE[torch.int32])},
            )
            return x.to(torch.int64)
        raise ValueError("unsupported DraftGraph operation: " + str(func))
