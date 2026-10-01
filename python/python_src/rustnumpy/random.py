import math as _math
import os as _os

from . import _core
from ._core import SeedSequence, asarray

_NativeGenerator = _core.Generator


def _entropy():
    return int.from_bytes(_os.urandom(8), "little")


class _BitGenerator:
    def __init__(self, gen):
        self._gen = gen
        self.seed_seq = None

    def random_raw(self, size=None):
        if size is None:
            return self._gen.random_raw()
        n = 1
        for s in (size if isinstance(size, tuple) else (size,)):
            n *= s
        return _core.array([self._gen.random_raw() for _ in range(n)], "uint64").reshape(size if isinstance(size, tuple) else (size,))

    def spawn(self, n):
        return [_BitGenerator(g._gen) for g in self._gen.spawn(n)]

    @property
    def state(self):
        return {"bit_generator": "PCG64"}


def _shape(size, *params):
    if size is None:
        shapes = [tuple(asarray(p).shape) for p in params]
        out = ()
        if shapes:
            from ._manip import broadcast_shapes

            out = broadcast_shapes(*shapes)
        return out
    return tuple(size) if isinstance(size, (tuple, list)) else (int(size),)


def _ret(x, size, params_scalar):
    x = asarray(x)
    if size is None and x.ndim == 0:
        return x.item()
    return x


class Generator:
    def __init__(self, bit_generator=None):
        if isinstance(bit_generator, Generator):
            self._g = bit_generator._g
        elif isinstance(bit_generator, _NativeGenerator):
            self._g = bit_generator
        elif isinstance(bit_generator, _BitGenerator):
            self._g = bit_generator._gen
        else:
            self._g = _NativeGenerator(_entropy() if bit_generator is None else int(bit_generator))

    def __repr__(self):
        return "Generator(PCG64)"

    @property
    def bit_generator(self):
        return _BitGenerator(self._g)

    @staticmethod
    def from_seed_sequence(seq):
        return Generator(_NativeGenerator.from_seed_sequence(seq))

    def spawn(self, n_children):
        return [Generator(g) for g in self._g.spawn(n_children)]

    def random_raw(self):
        return self._g.random_raw()

    def __getattr__(self, name):
        return getattr(self._g, name)

    def __dir__(self):
        return sorted(set(object.__dir__(self)) | {n for n in dir(self._g) if not n.startswith("_")})

    def random(self, size=None, dtype="float64", out=None):
        res = self._g.random(size)
        if dtype not in ("float64", float, _core.float64):
            res = asarray(res).astype(dtype)
        if out is not None:
            out[...] = res
            return out
        return res

    def _u(self, size):
        return asarray(self._g.random(size if size is not None and size != () else None))

    def standard_exponential(self, size=None, dtype="float64", method="zig", out=None):
        return _core.negative(_core.log1p(_core.negative(asarray(self._g.random(size)))))

    def standard_gamma(self, shape, size=None, dtype="float64", out=None):
        shp = asarray(shape)
        if shp.ndim == 0:
            return self._g.gamma(float(shp), 1.0, size)
        n = _shape(size, shp)
        flat = shp.reshape((-1,)) if False else _core.broadcast_to(shp, n).reshape((-1,)).tolist()
        return _core.array([self._g.gamma(float(s), 1.0, None) for s in flat]).reshape(n)

    def gamma(self, shape, scale=1.0, size=None):
        if asarray(shape).ndim == 0 and asarray(scale).ndim == 0:
            return self._g.gamma(float(shape), float(scale), size)
        return _core.multiply(self.standard_gamma(shape, size), scale)

    def chisquare(self, df, size=None):
        return _core.multiply(self.standard_gamma(_core.divide(asarray(df), 2.0), size), 2.0)

    def noncentral_chisquare(self, df, nonc, size=None):
        n = _shape(size, df, nonc)
        dfa = _core.broadcast_to(asarray(df), n).astype("float64")
        nca = _core.broadcast_to(asarray(nonc), n).astype("float64")
        pois = _core.array([self._g.poisson(float(l) / 2.0, None) if l > 0 else 0 for l in nca.reshape((-1,)).tolist()]).reshape(n)
        return _ret(_core.multiply(self.standard_gamma(_core.divide(_core.add(dfa, _core.multiply(2.0, pois)), 2.0), n if n else None), 2.0), size, False)

    def f(self, dfnum, dfden, size=None):
        num = _core.divide(self.chisquare(dfnum, size), dfnum)
        den = _core.divide(self.chisquare(dfden, size), dfden)
        return _ret(_core.divide(num, den), size, False)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        num = _core.divide(self.noncentral_chisquare(dfnum, nonc, size), dfnum)
        den = _core.divide(self.chisquare(dfden, size), dfden)
        return _ret(_core.divide(num, den), size, False)

    def standard_t(self, df, size=None):
        z = asarray(self._g.standard_normal(size))
        g = self.standard_gamma(_core.divide(asarray(df), 2.0), size)
        return _ret(_core.divide(z, _core.sqrt(_core.divide(_core.multiply(g, 2.0), df))), size, False)

    def standard_cauchy(self, size=None):
        return _ret(_core.tan(_core.multiply(_math.pi, _core.subtract(self._u(size), 0.5))), size, False)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        u = self._u(size)
        return _ret(_core.subtract(loc, _core.multiply(scale, _core.log(_core.negative(_core.log(u))))), size, False)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        u = _core.subtract(self._u(size), 0.5)
        val = _core.multiply(_core.sign(u), _core.log1p(_core.negative(_core.multiply(2.0, _core.absolute(u)))))
        return _ret(_core.subtract(loc, _core.multiply(scale, val)), size, False)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        u = self._u(size)
        return _ret(_core.add(loc, _core.multiply(scale, _core.log(_core.divide(u, _core.subtract(1.0, u))))), size, False)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return _ret(_core.exp(_core.add(mean, _core.multiply(sigma, asarray(self._g.standard_normal(size))))), size, False)

    def pareto(self, a, size=None):
        return _ret(_core.expm1(_core.divide(self.standard_exponential(size), a)), size, False)

    def power(self, a, size=None):
        e = self.standard_exponential(size)
        return _ret(_core.power(_core.negative(_core.expm1(_core.negative(e))), _core.divide(1.0, a)), size, False)

    def rayleigh(self, scale=1.0, size=None):
        return _ret(_core.multiply(scale, _core.sqrt(_core.multiply(2.0, self.standard_exponential(size)))), size, False)

    def weibull(self, a, size=None):
        return _ret(_core.power(self.standard_exponential(size), _core.divide(1.0, a)), size, False)

    def wald(self, mean, scale, size=None):
        mu = asarray(mean, "float64")
        n = asarray(self._g.standard_normal(size))
        y = _core.multiply(n, n)
        mu2 = _core.multiply(mu, mu)
        x = _core.add(mu, _core.divide(_core.multiply(mu2, y), _core.multiply(2.0, scale)))
        x = _core.subtract(x, _core.multiply(_core.divide(mu, _core.multiply(2.0, scale)), _core.sqrt(_core.add(_core.multiply(_core.multiply(4.0, mu), _core.multiply(scale, y)), _core.multiply(mu2, _core.multiply(y, y))))))
        u = self._u(size)
        return _ret(_core.where(_core.less_equal(u, _core.divide(mu, _core.add(mu, x))), x, _core.divide(mu2, x)), size, False)

    def triangular(self, left, mode, right, size=None):
        if not (left <= mode <= right) or left == right:
            raise ValueError("left > mode" if left > mode else ("mode > right" if mode > right else "left == right"))
        u = self._u(size)
        base = right - left
        lm = mode - left
        rm = right - mode
        fc = lm / base
        return _ret(_core.where(_core.less_equal(u, fc), _core.add(left, _core.sqrt(_core.multiply(u, _core.multiply(base, lm)))), _core.subtract(right, _core.sqrt(_core.multiply(_core.subtract(1.0, u), _core.multiply(base, rm))))), size, False)

    def geometric(self, p, size=None):
        u = self._u(size)
        if float(p) == 1.0:
            return _ret(_core.ones_like(u, "int64"), size, False)
        return _ret(_core.ceil(_core.divide(_core.log1p(_core.negative(u)), _math.log1p(-float(p)))).astype("int64"), size, False)

    def negative_binomial(self, n, p, size=None):
        n_out = _shape(size, n, p)
        g = self.gamma(n, _core.divide(_core.subtract(1.0, p), p), size)
        flat = asarray(g).reshape((-1,)).tolist() if asarray(g).ndim else [float(g)]
        res = _core.array([self._g.poisson(float(l), None) if l > 0 else 0 for l in flat], "int64")
        return _ret(res.reshape(n_out), size, False)

    def vonmises(self, mu, kappa, size=None):
        n = _shape(size, mu, kappa)
        out = []
        m = _core.broadcast_to(asarray(mu, "float64"), n).reshape((-1,)).tolist()
        k = _core.broadcast_to(asarray(kappa, "float64"), n).reshape((-1,)).tolist()
        for mi, ki in zip(m, k):
            if ki < 1e-8:
                out.append(_math.pi * (2 * float(self._g.random(None)) - 1))
                continue
            if ki < 1e-5:
                s = 1.0 / ki + ki
            else:
                r = 1 + _math.sqrt(1 + 4 * ki * ki)
                rho = (r - _math.sqrt(2 * r)) / (2 * ki)
                s = (1 + rho * rho) / (2 * rho)
            while True:
                u = float(self._g.random(None))
                v = float(self._g.random(None))
                z = _math.cos(_math.pi * u)
                w = (1 + s * z) / (s + z)
                y = ki * (s - w)
                vv = float(self._g.random(None))
                if y * (2 - y) - vv >= 0 or _math.log(y / vv) + 1 - y >= 0:
                    break
            res = _math.acos(w) if v >= 0.5 else -_math.acos(w)
            res = res + mi + _math.pi
            res = (res % (2 * _math.pi)) - _math.pi
            out.append(res)
        return _ret(_core.array(out).reshape(n), size, False)

    def zipf(self, a, size=None):
        n = _shape(size, a)
        av = _core.broadcast_to(asarray(a, "float64"), n).reshape((-1,)).tolist()
        out = []
        for ai in av:
            if ai <= 1.0:
                raise ValueError("a > 1")
            am1 = ai - 1.0
            b = 2.0**am1
            while True:
                u = 1.0 - float(self._g.random(None))
                v = float(self._g.random(None))
                x = _math.floor(u ** (-1.0 / am1))
                if x > 9.2233720368547758e18 or x < 1:
                    continue
                t = (1.0 + 1.0 / x) ** am1
                if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                    out.append(int(x))
                    break
        return _ret(_core.array(out, "int64").reshape(n), size, False)

    def logseries(self, p, size=None):
        n = _shape(size, p)
        pv = _core.broadcast_to(asarray(p, "float64"), n).reshape((-1,)).tolist()
        out = []
        for pi in pv:
            if not 0 < pi < 1:
                raise ValueError("p < 0, p >= 1 or p is NaN")
            r = _math.log1p(-pi)
            while True:
                v = float(self._g.random(None))
                if v >= pi:
                    out.append(1)
                    break
                u = float(self._g.random(None))
                q = -_math.expm1(r * u)
                if v <= q * q:
                    out.append(int(_math.floor(1 + _math.log(v) / _math.log(q))))
                    break
                if v >= q:
                    out.append(1)
                    break
                out.append(2)
                break
        return _ret(_core.array(out, "int64").reshape(n), size, False)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        n = _shape(size, ngood, nbad, nsample)
        g = _core.broadcast_to(asarray(ngood), n).reshape((-1,)).tolist()
        b = _core.broadcast_to(asarray(nbad), n).reshape((-1,)).tolist()
        s = _core.broadcast_to(asarray(nsample), n).reshape((-1,)).tolist()
        out = []
        for gi, bi, si in zip(g, b, s):
            if si > gi + bi:
                raise ValueError("ngood + nbad < nsample")
            good, bad, cnt = gi, bi, 0
            for _ in range(si):
                if float(self._g.random(None)) * (good + bad) < good:
                    cnt += 1
                    good -= 1
                else:
                    bad -= 1
            out.append(cnt)
        return _ret(_core.array(out, "int64").reshape(n), size, False)

    def multinomial(self, n, pvals, size=None):
        p = asarray(pvals, "float64")
        if float(_core.sum(p[:-1])) > 1.0 + 1e-12:
            raise ValueError("sum(pvals[:-1]) > 1.0")
        shape = _shape(size) if size is not None else ()
        count = 1
        for s in shape:
            count *= s
        rows = []
        pl = p.tolist()
        for _ in range(count):
            remaining, rest, row = int(n), 1.0, []
            for pi in pl[:-1]:
                if remaining > 0 and rest > 0:
                    x = self._g.binomial(remaining, min(1.0, max(0.0, pi / rest)), None)
                else:
                    x = 0
                row.append(int(x))
                remaining -= int(x)
                rest -= pi
            row.append(remaining)
            rows.append(row)
        res = _core.array(rows, "int64")
        return res.reshape(shape + (len(pl),)) if size is not None else res.reshape((len(pl),))

    def multivariate_hypergeometric(self, colors, nsample, size=None, method="marginals"):
        cl = [int(c) for c in asarray(colors).tolist()]
        shape = _shape(size) if size is not None else ()
        count = 1
        for s in shape:
            count *= s
        rows = []
        for _ in range(count):
            total, need, row = sum(cl), int(nsample), []
            for c in cl[:-1]:
                x = int(self.hypergeometric(c, total - c, need)) if need > 0 and total > 0 else 0
                row.append(x)
                need -= x
                total -= c
            row.append(need)
            rows.append(row)
        res = _core.array(rows, "int64")
        return res.reshape(shape + (len(cl),)) if size is not None else res.reshape((len(cl),))

    def permuted(self, x, axis=None, out=None):
        x = asarray(x)
        if axis is None:
            flat = x.reshape((-1,))
            keys = self._u((flat.size,))
            res = flat[_core.argsort(keys, 0)].reshape(tuple(x.shape))
        else:
            keys = self._u(tuple(x.shape))
            order = _core.argsort(keys, axis)
            from ._indexing import take_along_axis

            res = take_along_axis(x, order, axis)
        if out is not None:
            out[...] = res
            return out
        return res

    def bytes(self, length):
        words = [self._g.random_raw() for _ in range((length + 7) // 8)]
        return b"".join(w.to_bytes(8, "little") for w in words)[:length]

    def integers(self, low, high=None, size=None, dtype="int64", endpoint=False):
        dt = _core.dtype(dtype)
        if high is None:
            low, high = 0, low
        if dt.kind not in "iub":
            raise TypeError("Unsupported dtype %r for integers" % (dtype,))
        lo, hi = asarray(low), asarray(high)
        if lo.ndim == 0 and hi.ndim == 0:
            lo_i, hi_i = int(lo), int(hi) + (1 if endpoint else 0)
            info = _core.dtype("int64")
            if dt.name == "uint64" and (lo_i < 0 or hi_i > 2**64):
                raise ValueError("low is out of bounds for uint64" if lo_i < 0 else "high is out of bounds for uint64")
            if dt.kind == "i" and (lo_i < -(2 ** (8 * dt.itemsize - 1)) or hi_i > 2 ** (8 * dt.itemsize - 1)):
                raise ValueError("low is out of bounds for %s" % dt.name if lo_i < -(2 ** (8 * dt.itemsize - 1)) else "high is out of bounds for %s" % dt.name)
            if dt.name == "bool":
                lo_i, hi_i = max(lo_i, 0), min(hi_i, 2)
            if dt.name == "uint64" and hi_i > 2**63:
                raise _core.Unsupported("uint64 ranges above 2**63 are not supported")
            res = self._g.integers(lo_i, hi_i - 1 if False else hi_i, size, "int64", False) if False else self._g.integers(lo_i, hi_i, size, "int64", False)
            return asarray(res).astype(dt) if not (isinstance(res, int) and size is None) else (bool(res) if dt.name == "bool" else res)
        n = _shape(size, lo, hi)
        lo_b = _core.broadcast_to(lo.astype("int64"), n)
        hi_b = _core.broadcast_to(hi.astype("int64"), n)
        span = _core.subtract(_core.add(hi_b, 1 if endpoint else 0), lo_b)
        if bool(_core.any(_core.less_equal(span, 0))):
            raise ValueError("low >= high")
        u = asarray(self._g.random(n if n else None))
        res = _core.add(lo_b, _core.floor(_core.multiply(u, span)).astype("int64"))
        return res.astype(dt) if dt.name != "int64" else res

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        return self._g.choice(a, size, replace, p)


def default_rng(seed=None):
    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, SeedSequence):
        return Generator(_NativeGenerator.from_seed_sequence(seed))
    if seed is None:
        return Generator(_NativeGenerator(_entropy()))
    if isinstance(seed, _BitGenerator):
        return Generator(seed)
    return Generator(_NativeGenerator(int(seed)))
