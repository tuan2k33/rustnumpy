import contextlib
import math
import struct

_DEFAULTS = {
    "edgeitems": 3, "threshold": 1000, "floatmode": "maxprec", "precision": 8, "suppress": False, "linewidth": 75,
    "nanstr": "nan", "infstr": "inf", "sign": "-", "formatter": None, "legacy": False,
}
_opts = dict(_DEFAULTS)

_IMPLIED = {"int64", "float64", "complex128", "bool"}


def get_printoptions():
    return dict(_opts)


def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None,
                     formatter=None, sign=None, floatmode=None, *, legacy=None, override_repr=None):
    if floatmode is not None and floatmode not in ("fixed", "unique", "maxprec", "maxprec_equal"):
        raise ValueError("floatmode option must be one of fixed, unique, maxprec, maxprec_equal")
    if sign is not None and sign not in ("-", "+", " "):
        raise ValueError("sign option must be one of ' ', '+', or '-'")
    if precision is not None:
        if not isinstance(precision, int) or isinstance(precision, bool):
            raise TypeError("precision must be an integer")
        if precision < 0:
            raise ValueError("precision must be >= 0")
    updates = {
        "precision": precision, "threshold": threshold, "edgeitems": edgeitems, "linewidth": linewidth, "suppress": suppress,
        "nanstr": nanstr, "infstr": infstr, "formatter": formatter, "sign": sign, "floatmode": floatmode,
    }
    for key, value in updates.items():
        if value is not None:
            _opts[key] = value
    if legacy is not None:
        _opts["legacy"] = legacy


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    saved = get_printoptions()
    set_printoptions(*args, **kwargs)
    try:
        yield get_printoptions()
    finally:
        _opts.clear()
        _opts.update(saved)


def _round_trips(text, x, dtype_name):
    back = float(text)
    if dtype_name == "float64":
        return back == x
    if dtype_name == "float32":
        return struct.unpack("f", struct.pack("f", back))[0] == x
    return struct.unpack("e", struct.pack("e", back))[0] == x


def _shortest(x, dtype_name):
    if dtype_name == "float64":
        text = repr(x)
        return text
    for p in range(1, 18):
        text = "%.*e" % (p - 1, x)
        try:
            if _round_trips(text, x, dtype_name):
                return text
        except OverflowError:
            continue
    return repr(x)


def _digits(ax, dtype_name):
    text = _shortest(ax, dtype_name).lower()
    if "e" in text:
        mant, exp = text.split("e")
        exp = int(exp)
    else:
        mant, exp = text, 0
    ip, _, fp = mant.partition(".")
    all_digits = ip + fp
    point = len(ip) + exp
    stripped = all_digits.lstrip("0")
    point -= len(all_digits) - len(stripped)
    stripped = stripped.rstrip("0")
    if not stripped:
        return "0", 1
    return stripped, point


def _apply_trim(frac, trim):
    if trim == "k":
        return frac, True
    stripped = frac.rstrip("0")
    if trim == "0" and not stripped:
        return "0", True
    if trim == "-":
        return stripped, bool(stripped)
    return stripped, True


def _positional(x, dtype_name, precision, unique, trim, sign_plus, pad_left, pad_right, min_digits):
    neg = math.copysign(1.0, x) < 0
    ax = abs(x)
    if unique:
        digits, point = _digits(ax, dtype_name)
        if digits == "0":
            int_part, frac = "0", ""
        elif point <= 0:
            int_part, frac = "0", "0" * (-point) + digits
        elif point >= len(digits):
            int_part, frac = digits + "0" * (point - len(digits)), ""
        else:
            int_part, frac = digits[:point], digits[point:]
        if precision is not None and precision >= 0 and len(frac) > precision:
            int_part, _, frac = ("%.*f" % (precision, ax)).partition(".")
        if min_digits and len(frac) < min_digits:
            int_part, _, frac = ("%.*f" % (min_digits, ax)).partition(".")
    else:
        int_part, _, frac = ("%.*f" % (precision, ax)).partition(".")
    frac, keep_point = _apply_trim(frac, trim)
    head = ("-" if neg else "+" if sign_plus else "") + int_part
    if pad_left is not None:
        head = head.rjust(pad_left)
    tail = frac
    if pad_right is not None:
        tail = tail.ljust(pad_right)
    return head + ("." if keep_point else "") + tail


def _scientific(x, dtype_name, precision, unique, trim, sign_plus, pad_left, exp_digits, min_digits):
    neg = math.copysign(1.0, x) < 0
    ax = abs(x)
    if ax == 0 and unique:
        mant_digits, exp10 = "0", 0
    elif unique:
        digits, point = _digits(ax, dtype_name)
        mant_digits, exp10 = digits, point - 1
        if precision is not None and precision >= 0 and len(mant_digits) - 1 > precision:
            m, e = ("%.*e" % (precision, ax)).split("e")
            mant_digits, exp10 = m.replace(".", ""), int(e)
    else:
        m, e = ("%.*e" % (precision, ax)).split("e")
        mant_digits, exp10 = m.replace(".", ""), int(e)
    frac = mant_digits[1:]
    if unique and min_digits and len(frac) < min_digits and ax != 0:
        m, e = ("%.*e" % (min_digits, ax)).split("e")
        mant_digits, exp10 = m.replace(".", ""), int(e)
        frac = mant_digits[1:]
    elif unique and min_digits and len(frac) < min_digits:
        frac += "0" * (min_digits - len(frac))
    frac, keep_point = _apply_trim(frac, trim)
    head = ("-" if neg else "+" if sign_plus else "") + mant_digits[0]
    if pad_left is not None:
        head = head.rjust(pad_left)
    exp = "e%s%s" % ("-" if exp10 < 0 else "+", str(abs(exp10)).rjust(max(exp_digits or 2, 2), "0"))
    return head + ("." if keep_point else "") + frac + exp


class _FloatFormat:
    def __init__(self, values, dtype_name, opts, sign=None):
        self.dtype_name = dtype_name
        self.precision = opts["precision"]
        self.floatmode = opts["floatmode"]
        self.suppress = opts["suppress"]
        self.sign = sign or opts["sign"]
        self.nanstr = opts["nanstr"]
        self.infstr = opts["infstr"]
        self.exp_format = False
        self.unique = True
        self.trim = "."
        self.min_digits = None
        self.pad_left = 0
        self.pad_right = 0
        self.exp_size = -1
        finite = [v for v in values if math.isfinite(v)]
        non_zero = [abs(v) for v in finite if v != 0]
        if non_zero:
            max_val, min_val = max(non_zero), min(non_zero)
            if max_val >= 1.0e8 or (not self.suppress and (min_val < 0.0001 or max_val / min_val > 1000.0)):
                self.exp_format = True
        plus = self.sign == "+"
        if not finite:
            self.pad_left = self.pad_right = 0
            self.trim = "."
            self.exp_size = -1
            self.unique = True
            self.min_digits = None
        elif self.exp_format:
            trim, unique = ".", True
            if self.floatmode == "fixed":
                trim, unique = "k", False
            strs = [_scientific(v, dtype_name, self.precision, unique, trim, plus, None, None, None) for v in finite]
            frac_strs = [s.partition("e")[0] for s in strs]
            exp_strs = [s.partition("e")[2] for s in strs]
            int_part = [f.split(".")[0] for f in frac_strs]
            frac_part = [f.split(".")[1] if "." in f else "" for f in frac_strs]
            self.exp_size = max(len(e) for e in exp_strs) - 1
            self.trim = "k"
            self.precision = max(len(f) for f in frac_part)
            self.min_digits = self.precision
            self.unique = unique
            self.pad_left = max(len(i) for i in int_part)
            self.pad_right = self.exp_size + 2 + self.precision
        else:
            trim, unique = ".", True
            if self.floatmode == "fixed":
                trim, unique = "k", False
            strs = [_positional(v, dtype_name, self.precision, unique, trim, plus, None, None, None) for v in finite]
            int_part = [s.split(".")[0] for s in strs]
            frac_part = [s.split(".")[1] if "." in s else "" for s in strs]
            self.pad_left = max(len(i) for i in int_part)
            self.pad_right = max(len(f) for f in frac_part)
            self.exp_size = -1
            self.unique = unique
            if self.floatmode in ("fixed", "maxprec_equal"):
                self.precision = self.min_digits = self.pad_right
                self.trim = "k"
            else:
                self.trim = "."
                self.min_digits = 0
        if self.sign == " " and not any(math.copysign(1.0, v) < 0 for v in finite):
            self.pad_left += 1
        if len(finite) != len(values):
            neginf = self.sign != "-" or any(v < 0 for v in values if math.isinf(v))
            offset = self.pad_right + 1
            self.pad_left = max(self.pad_left, len(self.nanstr) - offset, len(self.infstr) + (1 if neginf else 0) - offset)

    def __call__(self, x):
        if not math.isfinite(x):
            if math.isnan(x):
                ret = ("+" if self.sign == "+" else "") + self.nanstr
            else:
                ret = ("-" if x < 0 else "+" if self.sign == "+" else "") + self.infstr
            return " " * (self.pad_left + self.pad_right + 1 - len(ret)) + ret
        plus = self.sign == "+"
        if self.exp_format:
            return _scientific(x, self.dtype_name, self.precision, self.unique, self.trim, plus, self.pad_left, self.exp_size, self.min_digits)
        return _positional(x, self.dtype_name, self.precision, self.unique, self.trim, plus, self.pad_left, self.pad_right, self.min_digits)


class _ComplexFormat:
    def __init__(self, values, dtype_name, opts):
        part = "float32" if dtype_name == "complex64" else "float64"
        self.real = _FloatFormat([v.real for v in values], part, opts)
        self.imag = _FloatFormat([v.imag for v in values], part, opts, sign="+")

    def __call__(self, x):
        r = self.real(x.real)
        i = self.imag(x.imag)
        sp = len(i.rstrip())
        return r + i[:sp] + "j" + i[sp:]


def _int_format(values, opts):
    if not values:
        return str
    sign = opts["sign"]
    hi, lo = max(values), min(values)
    hi_len = len(str(hi))
    if sign == " " and lo < 0:
        sign = "-"
    if hi >= 0 and sign in "+ ":
        hi_len += 1
    width = max(hi_len, len(str(lo)))
    spec = "{:%s%dd}" % (sign, width)
    return spec.format


def _element_formatter(flat_values, dtype_name, opts, is_scalar):
    user = opts.get("formatter")
    if dtype_name == "bool":
        true, false = ("True", "False") if is_scalar else (" True", "False")
        base = lambda x: true if x else false
        kind = "bool"
    elif dtype_name.startswith(("int", "uint")):
        base = _int_format(flat_values, opts)
        kind = "int"
    elif dtype_name.startswith("float"):
        base = _FloatFormat(flat_values, dtype_name, opts)
        kind = "float"
    else:
        base = _ComplexFormat(flat_values, dtype_name, opts)
        kind = "complexfloat"
    if user:
        chosen = user.get("all") or user.get(kind) or user.get({"int": "int_kind", "float": "float_kind", "complexfloat": "float_kind"}.get(kind, ""))
        if chosen is not None:
            return chosen
    return base


def _extend_line(s, line, word, line_width, next_line_prefix):
    needs_wrap = len(line) + len(word) > line_width
    if len(line) <= len(next_line_prefix):
        needs_wrap = False
    if needs_wrap:
        s += line.rstrip() + "\n"
        line = next_line_prefix
    return s, line + word


def _format_array(get, shape, gathered_shape, fmt, line_width, next_line_prefix, separator, edge_items, summary_insert):
    ndim = len(shape)

    def recurser(index, hanging_indent, curr_width):
        axis = len(index)
        axes_left = ndim - axis
        if axes_left == 0:
            return fmt(get(index))
        next_hanging_indent = hanging_indent + " "
        next_width = curr_width - len("]")
        a_len = shape[axis]
        show_summary = bool(summary_insert) and 2 * edge_items < a_len
        if show_summary:
            leading, trailing = edge_items, edge_items
        else:
            leading, trailing = 0, a_len

        def at_start(i):
            return index + (i,)

        def at_end(i):
            pos = (edge_items + (trailing - i)) if show_summary else a_len - i
            return index + (pos,)

        s = ""
        if axes_left == 1:
            elem_width = curr_width - max(len(separator.rstrip()), len("]"))
            line = hanging_indent
            for i in range(leading):
                s, line = _extend_line(s, line, recurser(at_start(i), next_hanging_indent, next_width), elem_width, hanging_indent)
                line += separator
            if show_summary:
                s, line = _extend_line(s, line, summary_insert, elem_width, hanging_indent)
                line += separator
            for i in range(trailing, 1, -1):
                s, line = _extend_line(s, line, recurser(at_end(i), next_hanging_indent, next_width), elem_width, hanging_indent)
                line += separator
            s, line = _extend_line(s, line, recurser(at_end(1), next_hanging_indent, next_width), elem_width, hanging_indent)
            s += line
        else:
            line_sep = separator.rstrip() + "\n" * (axes_left - 1)
            for i in range(leading):
                s += hanging_indent + recurser(at_start(i), next_hanging_indent, next_width) + line_sep
            if show_summary:
                s += hanging_indent + summary_insert + line_sep
            for i in range(trailing, 1, -1):
                s += hanging_indent + recurser(at_end(i), next_hanging_indent, next_width) + line_sep
            s += hanging_indent + recurser(at_end(1), next_hanging_indent, next_width)
        return "[" + s[len(hanging_indent):] + "]"

    return recurser((), next_line_prefix, line_width)


def _leading_trailing(a, edge):
    from . import _core

    data = a
    for ax in range(a.ndim):
        n = a.shape[ax]
        if n > 2 * edge:
            sel = _core.concatenate([_core.arange(0, edge), _core.arange(n - edge, n)])
            data = data[(slice(None),) * ax + (sel,)]
    return data


def _array2string(a, opts, separator, prefix, suffix, line_width):
    summary = a.size > opts["threshold"]
    edge = opts["edgeitems"]
    data = _leading_trailing(a, edge) if summary else a
    flat = data.reshape((-1,)).tolist()
    fmt = _element_formatter(flat, a.dtype.name, opts, a.ndim == 0)
    if a.ndim == 0:
        return fmt(flat[0])
    nested = data.tolist()

    def get(index):
        cur = nested
        for i in index:
            cur = cur[i]
        return cur

    next_line_prefix = " " + " " * len(prefix)
    return _format_array(get, tuple(a.shape), tuple(data.shape), fmt, line_width - len(suffix), next_line_prefix, separator, edge, "..." if summary else "")


def _resolve(opts_in, max_line_width, precision, suppress_small):
    opts = dict(_opts)
    if max_line_width is not None:
        opts["linewidth"] = max_line_width
    if precision is not None:
        opts["precision"] = precision
    if suppress_small is not None:
        opts["suppress"] = suppress_small
    return opts


def array2string(a, max_line_width=None, precision=None, suppress_small=None, separator=" ", prefix="", style=None, formatter=None,
                 threshold=None, edgeitems=None, sign=None, floatmode=None, suffix="", *, legacy=None):
    from . import _core

    a = _core.asarray(a)
    opts = _resolve(None, max_line_width, precision, suppress_small)
    for key, value in (("formatter", formatter), ("threshold", threshold), ("edgeitems", edgeitems), ("sign", sign), ("floatmode", floatmode)):
        if value is not None:
            opts[key] = value
    if a.size == 0:
        return "[]"
    return _array2string(a, opts, separator, prefix, suffix, opts["linewidth"])


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    from . import _core

    arr = _core.asarray(arr)
    opts = _resolve(None, max_line_width, precision, suppress_small)
    prefix = "array("
    lst = _array2string(arr, opts, ", ", prefix, ")", opts["linewidth"]) if arr.size else "[]"
    extras = []
    if (arr.size == 0 and tuple(arr.shape) != (0,)) or arr.size > opts["threshold"]:
        extras.append("shape=%s" % (tuple(arr.shape),))
    if arr.dtype.name not in _IMPLIED or arr.size == 0:
        extras.append("dtype=%s" % arr.dtype.name)
    if not extras:
        return prefix + lst + ")"
    arr_str = prefix + lst + ","
    extra_str = ", ".join(extras) + ")"
    last_line_len = len(arr_str) - (arr_str.rfind("\n") + 1)
    spacer = " "
    if last_line_len + len(extra_str) + 1 > opts["linewidth"]:
        spacer = "\n" + " " * len(prefix)
    return arr_str + spacer + extra_str


_SCI_ABOVE = {"float16": 1e3, "float32": 1e6, "float64": 1e16}


def _scalar_float_str(x, dtype_name, trim):
    if x != x:
        return "nan"
    if x in (float("inf"), float("-inf")):
        return "inf" if x > 0 else "-inf"
    sign = "-" if math.copysign(1.0, x) < 0 else ""
    ax = abs(x)
    if ax == 0:
        return sign + ("0" if trim else "0.0")
    mant, _, exp = _shortest(ax, dtype_name).lower().partition("e")
    exp10 = int(exp) if exp else 0
    if not exp:
        int_part, _, frac = mant.partition(".")
        digits = (int_part + frac).lstrip("0")
        exp10 = len(int_part.lstrip("0")) - 1 if int_part.strip("0") else -(len(frac) - len(frac.lstrip("0"))) - 1
    else:
        digits = mant.replace(".", "")
    digits = digits.rstrip("0") or "0"
    if ax >= _SCI_ABOVE[dtype_name] or ax < 1e-4:
        text = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        return "%s%se%s%02d" % (sign, text, "-" if exp10 < 0 else "+", abs(exp10))
    if exp10 >= 0:
        whole = digits[: exp10 + 1].ljust(exp10 + 1, "0")
        frac = digits[exp10 + 1 :]
    else:
        whole, frac = "0", "0" * (-exp10 - 1) + digits
    return sign + whole + ("." + frac if frac else ("" if trim else ".0"))


def _scalar_str(a):
    name = a.dtype.name
    value = a.item()
    if name in ("float16", "float32"):
        return _scalar_float_str(value, name, False)
    if name == "complex64":
        re_s = _scalar_float_str(value.real, "float32", True)
        im_s = _scalar_float_str(value.imag, "float32", True)
        if value.real == 0 and math.copysign(1.0, value.real) > 0:
            return im_s + "j"
        if not im_s.startswith("-"):
            im_s = "+" + im_s
        return "(%s%sj)" % (re_s, im_s)
    return str(value)


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    from . import _core

    a = _core.asarray(a)
    opts = _resolve(None, max_line_width, precision, suppress_small)
    if a.ndim == 0:
        return _scalar_str(a)
    if a.size == 0:
        return "[]"
    return _array2string(a, opts, " ", "", "", opts["linewidth"])
