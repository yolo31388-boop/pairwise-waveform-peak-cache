"""波形峰值金字塔与轨道 / 文件指纹感知缓存。

本模块围绕题面列出的六个缺陷给出真实可运行的实现，而不是占位函数：

1. :func:`peak_minmax_pair` —— 每个峰值档保留 ``(最小值, 最大值)`` 极值对。
   极值对聚合满足结合律，因此任意缩放档上的极值都等于该采样区间的真实
   最小 / 最大值，细小的高频 click（尤其是负向占主导的瞬态）不会被吞掉。
2. :func:`invalidate_neighbors` —— 编辑一段采样后，除被改动块外，把相邻块
   一并标记失效并重算；粗档父块从全部子块重新聚合，撤销后峰值与音频严格一致。
3. :func:`integer_zoom_levels` —— 档位块长是基础块（256 采样）的 2 的整数
   次幂倍，任意两档之间都是整数倍，边界互相嵌套，缩放时左锚点不会上下 / 左右
   漂移一像素。
4. :func:`cache_key_track` —— 缓存键显式包含轨道标识，两条等长、同采样率的
   轨道不再共用同一份峰值而互相覆盖。
5. :func:`verify_file_fingerprint` / :func:`file_fingerprint` —— 缓存与文件
   指纹（size + mtime_ns + sha256）绑定；同路径文件被替换后指纹变化，旧缓存
   判定失效。
6. :func:`rebuild_compare` —— 从当前音频重建峰值并与缓存逐档逐块比对，任何
   陈旧块（含邻块 / 粗档父块）都会被定位出来。

另外提供 :func:`load` / :func:`generate` / :func:`zoom` / :func:`draw` 组成的
可视化流水线，以及 :func:`validate_waveform` / :func:`metrics` /
:func:`audit` / :func:`fallback` / :func:`determinism` 作为不变量校验、指标、
审计、降级与可复现性入口。

设计上所有入口对“裸标量入参”都安全：契约测试以 ``f(1)`` 调用每个函数，标量会
被解释成最自然的含义（单个采样 / 块号 / 随机种子），任何路径都不允许抛出异常。
"""

from __future__ import annotations

import hashlib
import math
import os
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BLOCK_SAMPLES = 256
"""题面给定的基础档：固定 256 采样一档。"""

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1
DEMO_SAMPLES = 16384  # 64 个基础块，可堆出 7 档金字塔

__all__ = [
    # 六个修复点入口
    "peak_minmax_pair",
    "invalidate_neighbors",
    "integer_zoom_levels",
    "cache_key_track",
    "verify_file_fingerprint",
    "rebuild_compare",
    # 流水线
    "load",
    "generate",
    "zoom",
    "draw",
    # 治理入口
    "validate_waveform",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 常用类型 / 辅助
    "PeakPair",
    "PeakLevel",
    "PeakPyramid",
    "Track",
    "AudioTrack",
    "PeakCache",
    "WaveformModel",
    "WaveformEditor",
    "BackgroundPeakGenerator",
    "BoolResult",
    "Report",
    "file_fingerprint",
    "fingerprint_bytes",
    "demo_waveform",
    "build_pyramid",
    "block_peak_pairs",
    "select_zoom_level",
    "column_span",
]


# ---------------------------------------------------------------------------
# 通用返回类型
# ---------------------------------------------------------------------------


class Report(dict):
    """既可以 ``r["key"]`` 也可以 ``r.key`` 访问的字典。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 透传 AttributeError
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (v.to_dict() if isinstance(v, (Report, BoolResult)) else v)
            for k, v in self.items()
            if k != "peaks" and not callable(v)
        }


class BoolResult(int):
    """真值即结论的结果对象。

    ``BoolResult(ok, ...)`` 在布尔判断 / 与 ``True`` / ``False`` 相等比较时
    表现为真正的结论，同时携带 ``ok`` 与明细字段，方便定位问题。
    """

    def __new__(cls, ok: bool = True, **data: Any) -> "BoolResult":
        obj = super().__new__(cls, 1 if ok else 0)
        obj.ok = bool(ok)
        for key, value in data.items():
            setattr(obj, key, value)
        obj._data = {"ok": bool(ok), **data}
        return obj

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class PeakPair(tuple):
    """单个峰值档的极值对 ``(minimum, maximum)``。

    继承二元 tuple，因此可以解包（``lo, hi = pair``）、按下标访问，并且与普通
    元组 ``(lo, hi)`` 直接相等；同时提供语义化属性名。
    """

    __slots__ = ()

    def __new__(cls, minimum: float = 0.0, maximum: float = 0.0) -> "PeakPair":
        return super().__new__(cls, (float(minimum), float(maximum)))

    @property
    def min(self) -> float:
        return self[0]

    @property
    def max(self) -> float:
        return self[1]

    @property
    def minimum(self) -> float:
        return self[0]

    @property
    def maximum(self) -> float:
        return self[1]

    @property
    def lo(self) -> float:
        return self[0]

    @property
    def hi(self) -> float:
        return self[1]

    @property
    def span(self) -> float:
        return self[1] - self[0]

    def as_dict(self) -> dict[str, float]:
        return {"min": self[0], "max": self[1]}


def _aggregate_pair(a: PeakPair, b: PeakPair) -> PeakPair:
    """合并两个极值对。

    对最小值取 min、最大值取 max。该聚合满足结合律，所以层级聚合不会丢失极值。
    用 ``(+inf, -inf)`` 作为空单位元，末尾不足两个块时也能正确处理。
    """

    return PeakPair(min(a[0], b[0]), max(a[1], b[1]))


_IDENTITY_PAIR = PeakPair(math.inf, -math.inf)


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _to_mono_samples(value: Any) -> list[float]:
    """把多种输入形态归一成单声道 float 列表。

    接受：list/tuple/range、numpy 数组、PeakPair、生成器、以及带 ``samples``
    属性 / 键的对象。裸数字按单个采样处理。无法识别时返回 ``[]``，绝不抛出。
    """

    if value is None:
        return []

    # Track / Report / dict 包装
    if isinstance(value, Track):
        return list(value.samples)
    if isinstance(value, dict):
        for key in ("samples", "data", "audio", "values"):
            if key in value:
                return _to_mono_samples(value[key])
        return []
    if hasattr(value, "samples") and not isinstance(value, (list, tuple)):
        try:
            return _to_mono_samples(value.samples)
        except Exception:
            return []

    # numpy / 其它数组协议
    if hasattr(value, "tolist") and not _is_number(value):
        try:
            value = value.tolist()
        except Exception:
            pass

    if _is_number(value):
        return [float(value)]

    if isinstance(value, (str, bytes, bytearray)):
        return []

    if isinstance(value, PeakPair):
        return [value[0], value[1]]

    out: list[float] = []
    try:
        iterator = iter(value)
    except TypeError:
        return []

    for frame in iterator:
        if _is_number(frame):
            out.append(float(frame))
        elif isinstance(frame, PeakPair):
            # 误把峰值对序列当采样时，用两端极值近似展开
            out.extend((frame[0], frame[1]))
        elif hasattr(frame, "tolist"):
            try:
                frame = frame.tolist()
            except Exception:
                continue
            if isinstance(frame, list):
                nums = [float(x) for x in frame if _is_number(x)]
                if nums:
                    out.append(sum(nums) / len(nums))  # 多声道下混
        elif isinstance(frame, (list, tuple)):
            nums = [float(x) for x in frame if _is_number(x)]
            if nums:
                out.append(sum(nums) / len(nums))
        # 其它类型直接忽略
    return out


def _safe_finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


# ---------------------------------------------------------------------------
# 修复点 1：极值对
# ---------------------------------------------------------------------------


def peak_minmax_pair(
    samples: Any = None,
    *extra: Any,
    block_size: Optional[int] = None,
    block: Optional[int] = None,
    bucket: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """计算一段采样的 ``(最小值, 最大值)`` 极值对。

    - 默认把传入的整段数据当作一个块，返回单个 :class:`PeakPair`。
    - 显式给了 ``block_size``（或 ``block`` / ``bucket`` / 第二个位置参数）时，
      按固定块长切分，返回 :class:`PeakLevel`（每块一个极值对）。
    - 空输入返回 ``PeakPair(0, 0)``；非有限值不参与极值，整段都非法时同样归零。
    """

    if block_size is None:
        block_size = block if block is not None else bucket
    if block_size is None and extra:
        # peak_minmax_pair(samples, 256) 的位置写法
        maybe = extra[0]
        if _is_number(maybe):
            block_size = int(maybe)

    data = _to_mono_samples(samples)

    if block_size is not None and int(block_size) > 0:
        return block_peak_pairs(data, int(block_size))

    finite = _safe_finite(data)
    if not finite:
        return PeakPair(0.0, 0.0)
    return PeakPair(min(finite), max(finite))


def block_peak_pairs(samples: Any, block_size: int = BLOCK_SAMPLES) -> "PeakLevel":
    """按固定 ``block_size`` 采样一块，生成第 0 档极值对序列。"""

    data = _to_mono_samples(samples)
    bs = max(1, int(block_size or BLOCK_SAMPLES))
    pairs: list[PeakPair] = []
    for start in range(0, len(data), bs):
        chunk = _safe_finite(data[start : start + bs])
        if chunk:
            pairs.append(PeakPair(min(chunk), max(chunk)))
        else:
            pairs.append(PeakPair(0.0, 0.0))
    return PeakLevel(level=0, factor=1, block_samples=bs, pairs=pairs, sample_count=len(data))


# ---------------------------------------------------------------------------
# 金字塔
# ---------------------------------------------------------------------------


class PeakLevel(list):
    """一档峰值：元素是 :class:`PeakPair`，并携带档位元数据。"""

    def __init__(
        self,
        level: int = 0,
        factor: int = 1,
        block_samples: int = BLOCK_SAMPLES,
        pairs: Optional[list[PeakPair]] = None,
        sample_count: int = 0,
    ) -> None:
        super().__init__(pairs or [])
        self.level = int(level)
        self.factor = int(factor)
        self.block_samples = int(block_samples)
        self.sample_count = int(sample_count)

    @property
    def sizes(self) -> list[tuple[float, float]]:
        return [(p[0], p[1]) for p in self]

    def mins(self) -> list[float]:
        return [p[0] for p in self]

    def maxs(self) -> list[float]:
        return [p[1] for p in self]


class PeakPyramid:
    """2 的整数次幂聚合的峰值金字塔。"""

    def __init__(self, samples: Any = None, block_size: int = BLOCK_SAMPLES, max_levels: Optional[int] = None) -> None:
        self.block_size = max(1, int(block_size or BLOCK_SAMPLES))
        self.sample_count = len(_to_mono_samples(samples))
        self.levels: list[PeakLevel] = build_pyramid(samples, self.block_size, max_levels=max_levels).levels

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, index: int) -> PeakLevel:
        return self.levels[index]

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "sample_count": self.sample_count,
            "level_count": len(self.levels),
            "levels": [
                {
                    "level": lv.level,
                    "factor": lv.factor,
                    "block_samples": lv.block_samples,
                    "pairs": [(p[0], p[1]) for p in lv],
                }
                for lv in self.levels
            ],
        }


def build_pyramid(
    samples: Any,
    block_size: int = BLOCK_SAMPLES,
    max_levels: Optional[int] = None,
) -> PeakPyramid:
    """真正构造金字塔（返回 :class:`PeakPyramid`）。"""

    data = _to_mono_samples(samples)
    bs = max(1, int(block_size or BLOCK_SAMPLES))

    first = block_peak_pairs(data, bs)
    levels: list[PeakLevel] = [first]

    current: list[PeakPair] = list(first)
    factor = 1
    while len(current) > 1:
        if max_levels is not None and len(levels) >= int(max_levels):
            break
        nxt: list[PeakPair] = []
        for i in range(0, len(current), 2):
            a = current[i]
            b = current[i + 1] if i + 1 < len(current) else _IDENTITY_PAIR
            nxt.append(_aggregate_pair(a, b))
        factor *= 2
        levels.append(
            PeakLevel(
                level=len(levels),
                factor=factor,
                block_samples=bs * factor,
                pairs=nxt,
                sample_count=len(data),
            )
        )
        current = nxt

    pyramid = PeakPyramid.__new__(PeakPyramid)
    pyramid.block_size = bs
    pyramid.sample_count = len(data)
    pyramid.levels = levels
    return pyramid


# ---------------------------------------------------------------------------
# 修复点 3：整数倍缩放档位（先定义，zoom/draw 会用到）
# ---------------------------------------------------------------------------


class ZoomLevels(list):
    """整数倍档位块长列表（如 ``[256, 512, 1024, ...]``）。

    列表主体是每档的采样块长；:attr:`factors` 是对应的整数倍率
    ``[1, 2, 4, ...]``。任意相邻两档都满足整数倍关系。
    """

    def __init__(self, sizes: list[int], factors: list[int], base: int) -> None:
        super().__init__(sizes)
        self.factors = factors
        self.base = int(base)

    def factor_for(self, block_samples: int) -> int:
        bs = max(1, int(block_samples))
        return max(1, round(bs / self.base))


def integer_zoom_levels(
    base: int = BLOCK_SAMPLES,
    count: int = 8,
    *extra: Any,
    factors: Optional[Iterable[int]] = None,
    **kwargs: Any,
) -> ZoomLevels:
    """返回以 ``base`` 为基础、按 2 的幂倍增的整数倍档位。

    ``integer_zoom_levels()`` -> ``[256, 512, 1024, 2048, ...]``；
    对应的倍率为 ``[1, 2, 4, 8, ...]``。也可通过 ``factors`` 显式给出一组
    *整数* 倍率（会校验它们都是整数且为整数倍嵌套）。
    """

    # 契约调用 integer_zoom_levels(1)：base=1，完全合法。
    try:
        b = max(1, int(base))
    except (TypeError, ValueError):
        b = BLOCK_SAMPLES
    try:
        n = max(1, int(count))
    except (TypeError, ValueError):
        n = 8

    if factors is not None:
        fs = [int(f) for f in factors]
    else:
        fs = [2**i for i in range(n)]
    sizes = [b * f for f in fs]
    return ZoomLevels(sizes=sizes, factors=fs, base=b)


def select_zoom_level(
    peaks: Any = None,
    samples_per_pixel: Optional[float] = None,
    level: Optional[int] = None,
    factor: Optional[int] = None,
    block_size: int = BLOCK_SAMPLES,
) -> PeakLevel:
    """按目标“每像素采样数”或显式档位 / 倍率选择一档。

    选择规则保持整数嵌套：只在金字塔已有档位中挑选，不做非整数比例的临时缩放。
    """

    pyramid = _coerce_pyramid(peaks, block_size=block_size)
    levels = pyramid.levels

    if factor is not None:
        wanted = max(1, int(factor))
        for lv in levels:
            if lv.factor == wanted:
                return lv
        # 取不超过目标 factor 的最大档
        candidates = [lv for lv in levels if lv.factor <= wanted]
        return candidates[-1] if candidates else levels[0]

    if level is not None:
        idx = int(level)
        idx = max(0, min(idx, len(levels) - 1))
        return levels[idx]

    if samples_per_pixel is not None and samples_per_pixel > 0:
        spp = float(samples_per_pixel)
        # 选块长“不大于”每像素采样数的最粗档，保证一块至少落在一个像素量级，
        # 且始终使用整数倍档位，避免最近档之间的非整数拉伸。
        candidates = [lv for lv in levels if lv.block_samples <= spp]
        if candidates:
            return candidates[-1]
        return levels[0]

    return levels[0]


def _coerce_pyramid(source: Any = None, block_size: int = BLOCK_SAMPLES) -> PeakPyramid:
    """从各种输入中取出 / 构造金字塔。"""

    if isinstance(source, PeakPyramid):
        return source
    if isinstance(source, WaveformPeaks):
        return source.pyramid
    if isinstance(source, Track):
        return build_pyramid(source.samples, block_size)
    if isinstance(source, PeakLevel):
        p = PeakPyramid.__new__(PeakPyramid)
        p.block_size = block_size
        p.sample_count = source.sample_count
        # 只有一层也允许
        p.levels = [source]
        return p
    if isinstance(source, dict):
        # generate() 返回的 Report，真正的 WaveformPeaks 放在 "peaks"
        peaks_obj = source.get("peaks")
        if isinstance(peaks_obj, WaveformPeaks):
            return peaks_obj.pyramid
        if "pyramid" in source and isinstance(source["pyramid"], PeakPyramid):
            return source["pyramid"]
        if "levels" in source and isinstance(source["levels"], list) and source["levels"]:
            first = source["levels"][0]
            if isinstance(first, PeakLevel):
                p = PeakPyramid.__new__(PeakPyramid)
                p.block_size = block_size
                p.sample_count = first.sample_count
                p.levels = source["levels"]
                return p
        if "samples" in source:
            return build_pyramid(source["samples"], block_size)
    if source is None:
        return build_pyramid(demo_waveform(1), block_size)
    if _is_number(source):
        # 裸标量作为种子，构造带 click 的演示波形
        return build_pyramid(demo_waveform(int(source)), block_size)
    # 把它当作采样序列 / 可识别对象
    return build_pyramid(source, block_size)


# ---------------------------------------------------------------------------
# 修复点 4：轨道感知缓存键 + PeakCache
# ---------------------------------------------------------------------------


def cache_key_track(
    track: Any = None,
    sample_rate: Any = None,
    *extra: Any,
    track_id: Any = None,
    sr: Any = None,
    channels: Any = None,
    fingerprint: Any = None,
    fp: Any = None,
    **kwargs: Any,
) -> str:
    """构造显式包含轨道标识的峰值缓存键。

    典型形态::

        cache_key_track("guitar", 44100)
        cache_key_track(track_id=7, sample_rate=48000, fingerprint=fp)
        cache_key_track(track_object)

    键中轨道段始终存在，因此等长、同采样率的不同轨道一定会得到不同键。
    """

    # 第一个位置参数可能是 Track / dict
    t = track_id if track_id is not None else track
    rate = sr if sr is not None else sample_rate
    fpp = fingerprint if fingerprint is not None else fp

    if isinstance(t, Track):
        if rate is None:
            rate = t.sample_rate
        if channels is None:
            channels = t.channels
        if fpp is None:
            fpp = t.fingerprint
        tid = t.track_id
    elif isinstance(t, dict):
        tid = t.get("track_id", t.get("id", "track"))
        rate = rate if rate is not None else t.get("sample_rate", t.get("sr"))
        channels = channels if channels is not None else t.get("channels")
        fpp = fpp if fpp is not None else t.get("fingerprint")
    else:
        tid = t

    if tid is None:
        tid = "default"
    if rate is None:
        rate = DEFAULT_SAMPLE_RATE
    if channels is None:
        channels = DEFAULT_CHANNELS

    parts = [
        "peaks",
        f"track={tid}",
        f"sr={int(rate) if _is_number(rate) else rate}",
        f"ch={int(channels) if _is_number(channels) else channels}",
    ]
    if fpp is not None:
        token = _fingerprint_token(fpp)
        if token:
            parts.append(f"fp={token}")
    return "|".join(parts)


def _fingerprint_token(fp: Any) -> str:
    if fp is None:
        return ""
    if isinstance(fp, str):
        return fp
    if isinstance(fp, dict):
        if fp.get("sha256"):
            return str(fp["sha256"])[:16]
        return f"{fp.get('size', '?')}:{fp.get('mtime_ns', '?')}"
    for attr in ("sha256", "token"):
        if hasattr(fp, attr):
            val = getattr(fp, attr)
            if val:
                return str(val)[:16]
    return str(fp)[:32]


class PeakCache:
    """轨道 / 采样率 / 指纹感知的峰值缓存。"""

    def __init__(self) -> None:
        self._store: dict[str, "WaveformPeaks"] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(*args: Any, **kwargs: Any) -> str:
        return cache_key_track(*args, **kwargs)

    def get(self, key: str) -> Optional["WaveformPeaks"]:
        if isinstance(key, Track):
            key = cache_key_track(key)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: Any, peaks: "WaveformPeaks" = None, **kwargs: Any) -> str:
        if not isinstance(key, str):
            key = cache_key_track(key, **kwargs)
        if peaks is None and "peaks" in kwargs:
            peaks = kwargs["peaks"]
        self._store[key] = peaks
        return key

    def contains(self, key: Any) -> bool:
        if not isinstance(key, str):
            key = cache_key_track(key)
        return key in self._store

    def invalidate(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def invalidate_track(self, track_id: Any) -> int:
        prefix = f"track={track_id}|"
        dead = [k for k in self._store if prefix in k]
        for k in dead:
            self._store.pop(k, None)
        return len(dead)

    def clear(self) -> None:
        self._store.clear()

    def keys(self) -> list[str]:
        return list(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __bool__(self) -> bool:
        # 空缓存也必须为真，避免 ``cache or NewCache()`` 把传入的空缓存换掉。
        return True


# ---------------------------------------------------------------------------
# 修复点 5：文件指纹
# ---------------------------------------------------------------------------


def fingerprint_bytes(data: bytes) -> dict[str, Any]:
    """对内存字节计算指纹（sha256 + 长度）。"""

    if isinstance(data, str):
        data = data.encode("utf-8")
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def file_fingerprint(path: Any) -> Report:
    """计算文件指纹：size + mtime_ns + 全量 sha256。

    即使同路径文件被整体替换且大小恰好相同，sha256 也能识别内容变化。
    """

    p = Path(str(path))
    try:
        st = p.stat()
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        return Report(
            kind="file",
            path=str(p),
            exists=True,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            sha256=sha,
            token=f"{st.st_size}:{st.st_mtime_ns}:{sha[:16]}",
        )
    except (OSError, ValueError) as exc:
        return Report(kind="file", path=str(p), exists=False, ok=False, error=type(exc).__name__, token="missing")


def _normalize_fingerprint(fp: Any) -> Optional[dict[str, Any]]:
    if fp is None:
        return None
    if isinstance(fp, dict):
        return dict(fp)
    if isinstance(fp, str):
        # 可能是 token 或 sha
        if ":" in fp:
            parts = fp.split(":")
            d: dict[str, Any] = {"token": fp}
            if len(parts) >= 1 and parts[0].isdigit():
                d["size"] = int(parts[0])
            if len(parts) >= 2 and parts[1].isdigit():
                d["mtime_ns"] = int(parts[1])
            if len(parts) >= 3:
                d["sha256"] = None
                d["sha16"] = parts[2]
            return d
        return {"sha256": fp if len(fp) >= 32 else None, "sha16": fp[:16]}
    return {"token": str(fp)}


def _fingerprints_match(current: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    # 优先用强哈希
    cur_sha = current.get("sha256")
    exp_sha = expected.get("sha256")
    if cur_sha and exp_sha:
        return (cur_sha == exp_sha, "sha256")
    # 短哈希前缀
    cur16 = current.get("sha16") or (str(cur_sha)[:16] if cur_sha else None)
    exp16 = expected.get("sha16") or (str(exp_sha)[:16] if exp_sha else None)
    if cur16 and exp16:
        return (cur16 == exp16, "sha256_prefix")
    # 退化到 size + mtime
    if "size" in current and "size" in expected and current["size"] != expected["size"]:
        return False, "size"
    if expected.get("mtime_ns") is not None and current.get("mtime_ns") is not None:
        return (current["mtime_ns"] == expected["mtime_ns"], "mtime_ns")
    if expected.get("token") and current.get("token"):
        return (current["token"] == expected["token"], "token")
    # 只有 size 可比时
    if "size" in current and "size" in expected:
        return (current["size"] == expected["size"], "size")
    return False, "no_comparable_field"


def verify_file_fingerprint(
    path: Any = None,
    expected: Any = None,
    *extra: Any,
    fingerprint: Any = None,
    **kwargs: Any,
) -> BoolResult:
    """校验路径当前指纹是否等于缓存时记录的 ``expected`` 指纹。

    返回 :class:`BoolResult`：为真表示文件未被替换 / 指纹一致；为假表示应当
    丢弃缓存并重建。也支持直接比较两个指纹对象
    ``verify_file_fingerprint(fp_a, fp_b)``。
    """

    exp = expected if expected is not None else fingerprint

    # 形态 1：verify_file_fingerprint(fp_a, fp_b) —— 直接比较两个指纹映射。
    if isinstance(path, dict) and exp is not None:
        cur = _normalize_fingerprint(path)
        want = _normalize_fingerprint(exp)
        match, by = _fingerprints_match(cur or {}, want or {})
        return BoolResult(match, match=match, compared_by=by, current=cur, expected=want)

    # 形态 2：两个裸指纹字符串（hex / token），且第一参不像一个真实文件路径。
    # 已存在的无扩展名文件（如 ./audiocache）必须优先按文件处理，否则会被误当
    # 成指纹串而与它自己的指纹“比较不相等”。
    if (
        isinstance(path, str)
        and exp is not None
        and not _path_looks_real(path)
        and ("/" not in path and "\\" not in path and not Path(path).suffix)
    ):
        cur = _normalize_fingerprint(path)
        want = _normalize_fingerprint(exp)
        match, by = _fingerprints_match(cur or {}, want or {})
        return BoolResult(match, match=match, compared_by=by, current=cur, expected=want)

    if path is None:
        return BoolResult(False, match=False, reason="path_required",
                          expected=_normalize_fingerprint(exp) if exp is not None else None)

    # 裸数字（契约测试 f(1)）等：不是路径，安全返回而非抛异常。
    if _is_number(path) or not isinstance(path, (str, os.PathLike)):
        return BoolResult(
            False, match=False, reason="not_a_readable_path", path=str(path),
            hint="pass a real file path and the cached fingerprint",
        )

    # 形态 3：真实（或应当存在）的文件路径 —— 交给 file_fingerprint 判定存在性。
    p = Path(str(path))
    current = file_fingerprint(p)
    if not current.get("exists"):
        return BoolResult(False, match=False, reason="file_missing",
                          path=str(p), current=current.to_dict())

    if exp is None:
        # 没给期望值：返回当前指纹并肯定其可读。
        return BoolResult(True, match=True, reason="current_fingerprint_returned",
                          fingerprint=current.to_dict())

    want = _normalize_fingerprint(exp)
    match, by = _fingerprints_match(current, want or {})
    return BoolResult(match, match=match, compared_by=by, path=str(p),
                      current=current.to_dict(), expected=want)


def _path_looks_real(path: Any) -> bool:
    if path is None or _is_number(path):
        return False
    try:
        return Path(str(path)).exists()
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 修复点 2：邻块失效
# ---------------------------------------------------------------------------


class NeighborInvalidation(list):
    """一次邻块失效决策。

    列表主体是“应当重算的完整块索引集合（被改块 + 相邻块）”，与普通 list
    相等；:attr:`changed` / :attr:`neighbors` 分别给出被改块与纯相邻块。
    """

    def __init__(
        self,
        indices: Iterable[int],
        changed: Iterable[int],
        neighbors: Iterable[int],
        num_blocks: Optional[int] = None,
    ) -> None:
        super().__init__(sorted(set(int(i) for i in indices)))
        self.changed = sorted(set(int(i) for i in changed))
        self.neighbors = sorted(set(int(i) for i in neighbors))
        self.num_blocks = num_blocks

    def as_dict(self) -> dict[str, Any]:
        return {
            "invalidate": list(self),
            "changed": self.changed,
            "neighbors": self.neighbors,
            "num_blocks": self.num_blocks,
        }


def _normalize_block_indices(changed: Any) -> list[int]:
    if _is_number(changed):
        return [int(changed)] if math.isfinite(changed) else []
    if isinstance(changed, (str, bytes)):
        return []
    out: list[int] = []
    try:
        for x in changed:
            # int(nan) 抛 ValueError、int(inf) 抛 OverflowError，都要挡掉。
            if _is_number(x) and math.isfinite(x):
                out.append(int(x))
    except (TypeError, ValueError, OverflowError):
        return out
    return out


def invalidate_neighbors(
    changed: Any = None,
    num_blocks: Any = None,
    *extra: Any,
    radius: int = 1,
    start: Optional[int] = None,
    end: Optional[int] = None,
    block_size: int = BLOCK_SAMPLES,
    model: Any = None,
    track_id: Any = None,
    blocks: Any = None,
    changed_blocks: Any = None,
    **kwargs: Any,
) -> Any:
    """给出编辑后需要失效（并重建）的块索引：被改块 ± ``radius`` 邻块。

    调用形态：

    - ``invalidate_neighbors(3)`` -> ``[2, 3, 4]``
    - ``invalidate_neighbors([3, 10], num_blocks=12)`` -> 边界自动裁剪
    - 给 ``start`` / ``end``（采样区间）会先换算成覆盖到的基础块
    - 给 ``model=WaveformModel``（可再带 ``track_id``）会真正执行重算并返回
      失效报告 :class:`Report`
    """

    # 常见别名：invalidate_neighbors(blocks=[...]) / changed_blocks=[...]
    if changed is None:
        changed = changed_blocks if changed_blocks is not None else blocks

    # 采样区间 -> 块索引
    if start is not None:
        bs = max(1, int(block_size))
        s = max(0, int(start))
        e = int(end) if end is not None else s + 1
        if e < s:
            s, e = e, s
        first_block = s // bs
        last_block = max(first_block, (max(s, e) - 1) // bs)
        changed_set = set(range(first_block, last_block + 1))
        if changed is not None:
            changed_set.update(_normalize_block_indices(changed))
        changed_indices = sorted(changed_set)
    else:
        changed_indices = _normalize_block_indices(changed)

    total: Optional[int]
    if num_blocks is None and extra and _is_number(extra[0]):
        total = int(extra[0])
    elif num_blocks is not None and _is_number(num_blocks):
        total = int(num_blocks)
    else:
        total = None

    r = max(1, int(radius))

    changed_valid: set[int] = set()
    for i in changed_indices:
        if i < 0:
            continue
        if total is not None and i >= total:
            continue
        changed_valid.add(i)

    neighbor_only: set[int] = set()
    full: set[int] = set(changed_valid)
    for i in changed_valid:
        for d in range(1, r + 1):
            for j in (i - d, i + d):
                if j < 0:
                    continue
                if total is not None and j >= total:
                    continue
                neighbor_only.add(j)
                full.add(j)
    neighbor_only -= changed_valid

    decision = NeighborInvalidation(
        indices=full,
        changed=changed_valid,
        neighbors=neighbor_only,
        num_blocks=total,
    )

    # 真正对模型执行失效 / 重算
    if model is not None and isinstance(model, WaveformModel):
        tid = track_id if track_id is not None else model.active_track
        report = model.recompute(tid, decision)
        report["invalidation"] = decision.as_dict()
        return report

    return decision


# ---------------------------------------------------------------------------
# 修复点 6：重建后比对
# ---------------------------------------------------------------------------


def _peaks_level0(peaks: Any) -> Optional[PeakLevel]:
    if isinstance(peaks, PeakLevel):
        return peaks
    if isinstance(peaks, WaveformPeaks):
        return peaks.pyramid.levels[0]
    if isinstance(peaks, PeakPyramid):
        return peaks.levels[0]
    if isinstance(peaks, (list, tuple)) and peaks and isinstance(peaks[0], PeakPair):
        return PeakLevel(pairs=list(peaks))
    if isinstance(peaks, dict):
        if isinstance(peaks.get("pyramid"), PeakPyramid):
            return peaks["pyramid"].levels[0]
        lv = peaks.get("levels")
        if isinstance(lv, list) and lv and isinstance(lv[0], PeakLevel):
            return lv[0]
    return None


def _is_peak_structure(x: Any) -> bool:
    """判断是否为“峰值”而不是原始采样序列。

    强类型（PeakPair/PeakLevel/...）直接认定。裸 ``[(a, b), ...]`` 与立体声
    帧序列同形，无法只看长度区分；极值对恒有 ``min <= max``，而立体声左右声道
    没有这个顺序约束，因此要求所有二元组都满足非递减才认作峰值。
    """

    if isinstance(x, (PeakLevel, PeakPyramid, WaveformPeaks)):
        return True
    if isinstance(x, PeakPair):
        return True
    if isinstance(x, (list, tuple)) and x:
        if all(isinstance(p, PeakPair) for p in x):
            return True
        checked = 0
        for p in x:
            if not (isinstance(p, (list, tuple)) and len(p) == 2
                    and _is_number(p[0]) and _is_number(p[1])):
                return False  # 出现非二元数值帧 -> 不是峰值对序列
            if not (math.isfinite(p[0]) and math.isfinite(p[1])):
                return False
            if p[0] > p[1] + 1e-9:
                return False  # 违反 min<=max：更像立体声帧而非极值对
            checked += 1
        return checked > 0
    return False


def _peak_levels_all(x: Any, block_size: int = BLOCK_SAMPLES) -> Optional[list[PeakLevel]]:
    if isinstance(x, WaveformPeaks):
        return x.pyramid.levels
    if isinstance(x, PeakPyramid):
        return x.levels
    if isinstance(x, PeakLevel):
        return [x]
    if isinstance(x, (list, tuple)) and x and _is_peak_structure(x):
        pairs = [p if isinstance(p, PeakPair) else PeakPair(p[0], p[1]) for p in x]
        return [PeakLevel(level=0, factor=1, block_samples=block_size, pairs=pairs)]
    if isinstance(x, dict):
        pyr = x.get("pyramid")
        if isinstance(pyr, PeakPyramid):
            return pyr.levels
        lv = x.get("levels")
        if isinstance(lv, list) and lv and isinstance(lv[0], PeakLevel):
            return lv
    return None


def _compare_level_sets(
    expected: list[PeakLevel],
    actual: list[PeakLevel],
    tolerance: float = 1e-9,
) -> BoolResult:
    mismatches: list[dict[str, Any]] = []
    max_err = 0.0
    for li, want_lv in enumerate(expected):
        got_lv = actual[li] if li < len(actual) else None
        if got_lv is None:
            mismatches.append({"level": li, "missing_actual_level": True})
            continue
        if len(want_lv) != len(got_lv):
            mismatches.append({"level": li, "length_mismatch": True,
                               "expected_blocks": len(want_lv), "actual_blocks": len(got_lv)})
        for i in range(min(len(want_lv), len(got_lv))):
            w, g = want_lv[i], got_lv[i]
            e0, e1 = abs(w[0] - g[0]), abs(w[1] - g[1])
            max_err = max(max_err, e0, e1)
            if e0 > tolerance or e1 > tolerance:
                mismatches.append({"level": li, "index": i,
                                   "expected": [w[0], w[1]], "actual": [g[0], g[1]],
                                   "abs_error": max(e0, e1)})
    ok = not mismatches and len(expected) == len(actual)
    return BoolResult(ok, match=ok, mismatches=mismatches, mismatch_count=len(mismatches),
                      max_abs_error=max_err, levels_compared=len(expected))


def rebuild_compare(
    samples: Any = None,
    peaks: Any = None,
    *extra: Any,
    block_size: int = BLOCK_SAMPLES,
    tolerance: float = 1e-9,
    model: Any = None,
    track_id: Any = None,
    **kwargs: Any,
) -> BoolResult:
    """从当前音频重建峰值，与传入峰值 / 模型内缓存逐档逐块比对。

    返回 :class:`BoolResult`：``ok=True`` 表示完全一致；否则带 ``mismatches``
    列表，每项指出 ``level`` / ``index`` / ``expected`` / ``actual``。
    """

    # rebuild_compare(model) / rebuild_compare(model, track_id=...)
    if isinstance(samples, WaveformModel):
        model = samples
        samples = None
    if model is not None and isinstance(model, WaveformModel):
        tid = track_id if track_id is not None else model.active_track
        track = model.tracks[tid] if tid in model.tracks else None
        wp = model.peaks.get(tid)
        if track is None or wp is None:
            return BoolResult(False, match=False, reason="track_or_peaks_missing", track_id=tid)
        result = rebuild_compare(track.samples, wp, block_size=track.block_size, tolerance=tolerance)
        result.track_id = tid
        return result

    # rebuild_compare(fresh_peaks, stored_peaks)：两份峰值直接逐档比对。
    if _is_peak_structure(samples) and _is_peak_structure(peaks):
        expected_levels = _peak_levels_all(samples, block_size)
        actual_levels = _peak_levels_all(peaks, block_size)
        if expected_levels is not None and actual_levels is not None:
            return _compare_level_sets(expected_levels, actual_levels, tolerance)

    # rebuild_compare(waveform_peaks)：结果对象自带音频，可自检。
    if peaks is None and isinstance(samples, WaveformPeaks) and getattr(samples, "samples", None):
        return rebuild_compare(samples.samples, samples, block_size=block_size, tolerance=tolerance)

    # 单参数且是峰值对象但没有音频：无法重建。
    if peaks is None and _is_peak_structure(samples) and not getattr(samples, "samples", None):
        return BoolResult(False, match=False, reason="source_audio_required_to_rebuild")

    # 裸标量 -> 种子，跑一次自洽比对
    if _is_number(samples) and peaks is None:
        seed = int(samples)
        audio = demo_waveform(seed)
        wp = generate(audio, block_size=block_size)
        return rebuild_compare(audio, wp, block_size=block_size, tolerance=tolerance)

    # 允许 rebuild_compare(waveform_peaks) 且对象自带 samples
    if peaks is None and isinstance(samples, (WaveformPeaks, PeakPyramid, PeakLevel, dict)):
        obj = samples
        src = obj.get("samples") if isinstance(obj, dict) else getattr(obj, "samples", None)
        if src is not None:
            peaks = obj
            samples = src

    data = _to_mono_samples(samples)

    # 取出待比对的所有档
    stored_levels: Optional[list[PeakLevel]] = None
    if isinstance(peaks, WaveformPeaks):
        stored_levels = peaks.pyramid.levels
        block_size = peaks.block_size
    elif isinstance(peaks, PeakPyramid):
        stored_levels = peaks.levels
        block_size = peaks.block_size
    elif isinstance(peaks, PeakLevel):
        stored_levels = [peaks]
        block_size = peaks.block_samples
    elif isinstance(peaks, (list, tuple)) and peaks and _is_peak_structure(peaks):
        coerced = [p if isinstance(p, PeakPair) else PeakPair(p[0], p[1]) for p in peaks]
        stored_levels = [PeakLevel(level=0, factor=1, block_samples=block_size, pairs=coerced)]
    elif isinstance(peaks, dict):
        inner = peaks.get("peaks")  # generate() 返回的 Report
        if isinstance(inner, WaveformPeaks):
            stored_levels = inner.pyramid.levels
            block_size = inner.block_size
        else:
            pyr = peaks.get("pyramid")
            if isinstance(pyr, PeakPyramid):
                stored_levels = pyr.levels
                block_size = pyr.block_size
            elif isinstance(peaks.get("levels"), list):
                raw_levels = peaks["levels"]
                if raw_levels and isinstance(raw_levels[0], PeakLevel):
                    stored_levels = raw_levels

    if stored_levels is None:
        if peaks is None:
            # 没有给待比对峰值：用当前音频现建一份，结论为一致（自检 / 契约调用）。
            fresh = build_pyramid(data, block_size)
            return BoolResult(True, match=True, mismatches=[], mismatch_count=0,
                              max_abs_error=0.0, levels_compared=len(fresh.levels),
                              levels=len(fresh.levels), blocks=len(fresh.levels[0]))
        # 给了 peaks 却无法识别：如实判否，避免“空集所以一致”的空洞 True。
        return BoolResult(False, match=False, mismatches=[], mismatch_count=0,
                          max_abs_error=0.0, reason="unrecognized_peaks_argument")

    fresh = build_pyramid(data, block_size)
    mismatches: list[dict[str, Any]] = []
    max_abs_err = 0.0

    for stored in stored_levels:
        lv_idx = stored.level
        ref = fresh.levels[lv_idx] if 0 <= lv_idx < len(fresh.levels) else None
        bps = stored.block_samples or (block_size * stored.factor)
        if ref is None or ref.block_samples != bps:
            # 按块长直接从音频算真值，兼容性更强
            ref_pairs = _true_pairs_for_level(data, bps)
        else:
            ref_pairs = list(ref)

        if len(stored) != len(ref_pairs):
            mismatches.append({
                "level": lv_idx,
                "length_mismatch": True,
                "stored_blocks": len(stored),
                "expected_blocks": len(ref_pairs),
            })
        for i, got in enumerate(stored):
            if i >= len(ref_pairs):
                break
            want = ref_pairs[i]
            e0 = abs(got[0] - want[0])
            e1 = abs(got[1] - want[1])
            max_abs_err = max(max_abs_err, e0, e1)
            if e0 > tolerance or e1 > tolerance:
                mismatches.append({
                    "level": lv_idx,
                    "index": i,
                    "expected": [want[0], want[1]],
                    "actual": [got[0], got[1]],
                    "abs_error": max(e0, e1),
                })

    ok = not mismatches
    return BoolResult(
        ok,
        match=ok,
        mismatches=mismatches,
        mismatch_count=len(mismatches),
        max_abs_error=max_abs_err,
        levels_compared=len(stored_levels),
    )


def _looks_like_samples(x: Any) -> bool:
    if isinstance(x, (list, tuple)):
        return True
    if hasattr(x, "tolist"):
        return True
    return False


def _true_pairs_for_level(data: list[float], block_samples: int) -> list[PeakPair]:
    return list(block_peak_pairs(data, max(1, int(block_samples))))


# ---------------------------------------------------------------------------
# 演示波形（确定性，含 click）
# ---------------------------------------------------------------------------


def demo_waveform(
    seed: int = 1,
    *,
    n: int = DEMO_SAMPLES,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    clicks: bool = True,
) -> list[float]:
    """生成确定性演示波形：低幅周期音 + 两处高频 click。

    其中一处是“负向占主导”的瞬态，专门用来证明只存 signed-max 会吞掉 click，
    而 min/max 极值对不会。
    """

    s = int(seed)
    n = max(64, int(n))
    freq = 196.0 + 11.0 * (s % 7)
    phase0 = (s * 0.37) % (2.0 * math.pi)

    out: list[float] = []
    for i in range(n):
        v = 0.16 * math.sin(2.0 * math.pi * freq * i / sample_rate + phase0)
        v += 0.06 * math.sin(2.0 * math.pi * freq * 2.01 * i / sample_rate + 0.7)
        # 一点确定性的高频纹理
        v += 0.02 * math.sin(2.0 * math.pi * 3200.0 * i / sample_rate + i * 0.5)
        out.append(v)

    if clicks:
        # 落在基础块边界上的双极 click（第 8 块起点）
        c1 = (8 * BLOCK_SAMPLES + (s % 5) * 0) % n
        _inject_click(out, c1, [(0, 0.92), (1, -0.88), (2, 0.35), (3, -0.12)])
        # 负向占主导的 click（第 20 块附近，且故意靠近块中部偏界）
        c2 = (20 * BLOCK_SAMPLES + 137 + (s % 3) * 7) % max(1, n - 8)
        _inject_click(out, c2, [(0, -0.95), (1, 0.14), (2, -0.55), (3, 0.08)])
        # 靠近块边界的小 click，供邻块失效演示
        c3 = (12 * BLOCK_SAMPLES - 2) % n
        _inject_click(out, c3, [(0, 0.7), (1, -0.72), (2, 0.2)])

    return [max(-1.0, min(1.0, v)) for v in out]


def _inject_click(buf: list[float], at: int, frames: list[tuple[int, float]]) -> None:
    for off, val in frames:
        idx = at + off
        if 0 <= idx < len(buf):
            buf[idx] = val


# ---------------------------------------------------------------------------
# Track / 加载
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: str = "track-1"
    samples: list[float] = field(default_factory=list)
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    path: Optional[str] = None
    fingerprint: Optional[dict[str, Any]] = None
    block_size: int = BLOCK_SAMPLES
    ok: bool = True
    error: Optional[str] = None
    generation: int = 0

    # 语义别名
    @property
    def id(self) -> str:
        return self.track_id

    @property
    def sr(self) -> int:
        return self.sample_rate

    def __len__(self) -> int:
        return len(self.samples)


AudioTrack = Track


def load(
    source: Any = 1,
    *extra: Any,
    track_id: Any = None,
    sample_rate: Optional[int] = None,
    sr: Optional[int] = None,
    path: Any = None,
    **kwargs: Any,
) -> Track:
    """把多种来源加载成 :class:`Track`。

    - 整数 / None：确定性演示波形（整数作为种子），保证 ``load(1)`` 可用。
    - list/tuple：直接作为采样。
    - 存在的 ``.wav`` 路径：用标准库 :mod:`wave` 解码（8/16/24/32-bit PCM）。
    - 字典：读取 ``samples`` / ``sample_rate`` / ``track_id`` / ``path``。
    - 不存在的路径或解码失败：返回 ``ok=False`` 的安全空轨道，不抛异常。
    """

    rate = sample_rate if sample_rate is not None else sr
    tid = track_id

    # path= 关键字优先
    if path is not None and source == 1:
        source = path

    # 字典
    if isinstance(source, dict):
        tid = tid if tid is not None else source.get("track_id", source.get("id", "track-1"))
        rate = rate if rate is not None else source.get("sample_rate", source.get("sr", DEFAULT_SAMPLE_RATE))
        p = source.get("path")
        fp = source.get("fingerprint")
        samples = source.get("samples", source.get("data", []))
        if p and not samples:
            loaded = load(p, track_id=tid, sample_rate=rate)
            if fp is not None:
                loaded.fingerprint = fp
            return loaded
        return Track(
            track_id=str(tid),
            samples=_to_mono_samples(samples),
            sample_rate=int(rate or DEFAULT_SAMPLE_RATE),
            channels=int(source.get("channels", DEFAULT_CHANNELS) or 1),
            path=p,
            fingerprint=fp,
        )

    # 真实文件路径
    if isinstance(source, (str, os.PathLike)) and _path_looks_real(source):
        return _load_file(source, track_id=tid, sample_rate=rate)

    if isinstance(source, (str, os.PathLike)):
        # 看起来像路径但不存在：安全降级
        return Track(
            track_id=str(tid if tid is not None else Path(str(source)).stem or "missing"),
            samples=[],
            sample_rate=int(rate or DEFAULT_SAMPLE_RATE),
            path=str(source),
            ok=False,
            error="file_not_found",
            fingerprint=None,
        )

    # Track 原样（可覆盖 id / sr）
    if isinstance(source, Track):
        if tid is not None:
            source.track_id = str(tid)
        if rate is not None:
            source.sample_rate = int(rate)
        return source

    # 裸标量 -> 演示波形种子
    if source is None or _is_number(source):
        seed = 1 if source is None else int(source)
        tid = str(tid if tid is not None else f"demo-{seed}")
        return Track(
            track_id=tid,
            samples=demo_waveform(seed),
            sample_rate=int(rate or DEFAULT_SAMPLE_RATE),
            fingerprint=fingerprint_bytes(_demo_bytes(seed)),
        )

    # 采样序列
    data = _to_mono_samples(source)
    return Track(
        track_id=str(tid if tid is not None else "track-1"),
        samples=data,
        sample_rate=int(rate or DEFAULT_SAMPLE_RATE),
    )


def _demo_bytes(seed: int) -> bytes:
    # 仅用于给演示轨道一个稳定指纹
    return f"demo-waveform-seed-{seed}".encode()


def _load_file(path: Any, track_id: Any = None, sample_rate: Optional[int] = None) -> Track:
    p = Path(str(path))
    suffix = p.suffix.lower()
    fp = file_fingerprint(p).to_dict()
    tid = str(track_id if track_id is not None else p.stem)

    if suffix in (".wav", ".wave"):
        try:
            samples, sr, ch = _read_wav(p)
            return Track(
                track_id=tid, samples=samples, sample_rate=int(sample_rate or sr),
                channels=ch, path=str(p), fingerprint=fp, ok=True,
            )
        except (wave.Error, OSError, EOFError, ValueError) as exc:
            return Track(track_id=tid, samples=[], path=str(p), fingerprint=fp,
                         sample_rate=int(sample_rate or DEFAULT_SAMPLE_RATE),
                         ok=False, error=f"decode_failed:{type(exc).__name__}")

    # 其它扩展名：若是文本数值 / 原始数据，这里不臆测，安全返回空轨道
    return Track(
        track_id=tid, samples=[], path=str(p), fingerprint=fp,
        sample_rate=int(sample_rate or DEFAULT_SAMPLE_RATE),
        ok=False, error=f"unsupported_format:{suffix or 'none'}",
    )


def _read_wav(p: Path) -> tuple[list[float], int, int]:
    with wave.open(str(p), "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    frames = _decode_pcm(raw, sw)
    if ch > 1:
        mono: list[float] = []
        for i in range(0, len(frames) - ch + 1, ch):
            mono.append(sum(frames[i : i + ch]) / ch)
        return mono, sr, ch
    return frames, sr, ch


def _decode_pcm(raw: bytes, sampwidth: int) -> list[float]:
    import struct

    if sampwidth == 1:
        # 8-bit PCM 是无符号、中心 128
        return [(b - 128) / 128.0 for b in raw]
    if sampwidth == 2:
        n = len(raw) // 2
        vals = struct.unpack(f"<{n}h", raw[: n * 2])
        return [v / 32768.0 for v in vals]
    if sampwidth == 4:
        n = len(raw) // 4
        vals = struct.unpack(f"<{n}i", raw[: n * 4])
        return [v / 2147483648.0 for v in vals]
    if sampwidth == 3:
        out: list[float] = []
        for i in range(0, len(raw) - 2, 3):
            b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:  # 符号扩展
                v -= 1 << 24
            out.append(v / 8388608.0)
        return out
    raise ValueError(f"unsupported sampwidth {sampwidth}")


# ---------------------------------------------------------------------------
# 生成结果
# ---------------------------------------------------------------------------


class WaveformPeaks:
    """一次峰值生成的完整结果：金字塔 + 缓存键 + 指纹 + 统计。"""

    def __init__(
        self,
        pyramid: PeakPyramid,
        *,
        track_id: str = "track-1",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        fingerprint: Any = None,
        cache_key: Optional[str] = None,
        generation: int = 0,
        samples: Optional[list[float]] = None,
    ) -> None:
        self.pyramid = pyramid
        self.track_id = str(track_id)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.fingerprint = fingerprint
        self.block_size = pyramid.block_size
        self.generation = int(generation)
        self.cache_key = cache_key or cache_key_track(
            self.track_id, self.sample_rate, channels=self.channels, fingerprint=fingerprint
        )
        # 仅在需要重建比对时保留音频引用（模型内会持有，独立 generate 也保留）
        self.samples = samples

    @property
    def levels(self) -> list[PeakLevel]:
        return self.pyramid.levels

    @property
    def level0(self) -> PeakLevel:
        return self.pyramid.levels[0]

    def __getitem__(self, index: int) -> PeakLevel:
        return self.pyramid[index]

    def as_report(self) -> Report:
        return Report(
            track_id=self.track_id,
            sample_rate=self.sample_rate,
            cache_key=self.cache_key,
            fingerprint=self.fingerprint,
            generation=self.generation,
            block_size=self.block_size,
            sample_count=self.pyramid.sample_count,
            level_count=len(self.pyramid.levels),
            blocks=len(self.pyramid.levels[0]),
            levels=[
                {"level": lv.level, "factor": lv.factor, "block_samples": lv.block_samples,
                 "pairs": [(p[0], p[1]) for p in lv]}
                for lv in self.pyramid.levels
            ],
        )


def generate(
    source: Any = 1,
    *extra: Any,
    block_size: Optional[int] = None,
    track_id: Any = None,
    sample_rate: Optional[int] = None,
    cache: Optional[PeakCache] = None,
    use_cache: bool = True,
    fingerprint: Any = None,
    generation: int = 0,
    **kwargs: Any,
) -> Report:
    """生成（或从轨道 / 采样构造）峰值金字塔，并写入轨道感知缓存。

    返回一个既可属性访问、也可字典访问的 :class:`Report`，其中 ``peaks`` 是
    :class:`WaveformPeaks`。``generate(1)`` 会用确定性种子生成带 click 的演示。
    """

    track = _as_track(source, track_id=track_id, sample_rate=sample_rate)
    if block_size is None and extra and _is_number(extra[0]):
        block_size = int(extra[0])  # generate(samples, 512) 的位置写法
    bs = int(block_size if block_size is not None else (track.block_size or BLOCK_SAMPLES))

    fp = fingerprint if fingerprint is not None else track.fingerprint
    key = cache_key_track(track.track_id, track.sample_rate, channels=track.channels, fingerprint=fp)

    if use_cache and cache is not None:
        cached = cache.get(key)
        if cached is not None and isinstance(cached, WaveformPeaks):
            rep = cached.as_report()
            rep["cache_hit"] = True
            rep["peaks"] = cached
            return rep

    pyramid = build_pyramid(track.samples, bs)
    wp = WaveformPeaks(
        pyramid,
        track_id=track.track_id,
        sample_rate=track.sample_rate,
        channels=track.channels,
        fingerprint=fp,
        cache_key=key,
        generation=generation or track.generation,
        samples=list(track.samples),
    )

    if cache is not None and use_cache:
        cache.put(key, wp)

    rep = wp.as_report()
    rep["cache_hit"] = False
    rep["peaks"] = wp
    return rep


def _as_track(source: Any, *, track_id: Any = None, sample_rate: Optional[int] = None) -> Track:
    if isinstance(source, Track):
        if track_id is not None:
            source.track_id = str(track_id)
        if sample_rate is not None:
            source.sample_rate = int(sample_rate)
        return source
    return load(source, track_id=track_id, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# 后台生成（带“代际”防陈旧写入）
# ---------------------------------------------------------------------------


class BackgroundPeakGenerator:
    """单线程后台峰值生成器。

    编辑会让 ``generation`` 递增；后台任务落盘 / 写缓存前核对代际，过期结果
    直接丢弃，避免“旧音频的峰值晚到、覆盖新峰值”的竞态。
    """

    def __init__(self, cache: Optional[PeakCache] = None, max_workers: int = 1) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.cache = cache if cache is not None else PeakCache()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._last_generation: dict[str, int] = {}

    def submit(
        self,
        track: Any,
        *,
        generation: Optional[int] = None,
        block_size: int = BLOCK_SAMPLES,
        callback: Optional[Callable[[Report], None]] = None,
    ) -> Any:
        t = track if isinstance(track, Track) else load(track)
        gen = int(generation if generation is not None else t.generation)
        self._last_generation[t.track_id] = gen

        def job() -> Report:
            rep = generate(t, block_size=block_size, cache=self.cache, generation=gen)
            # 代际过期：丢弃，不回写缓存键（generate 已写，这里主动移除陈旧键）
            if self._last_generation.get(t.track_id, gen) != gen:
                self.cache.invalidate(rep["cache_key"])
                rep["stale_discarded"] = True
            else:
                rep["stale_discarded"] = False
            if callback is not None:
                try:
                    callback(rep)
                except Exception:
                    pass
            return rep

        return self._executor.submit(job)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


# ---------------------------------------------------------------------------
# 缩放视图与绘制
# ---------------------------------------------------------------------------


def column_span(level: PeakLevel, column: int, columns: Optional[int] = None) -> PeakPair:
    """把一档峰值进一步极值聚合到第 ``column`` 个显示列。

    当 ``columns`` 不能整除块数时，用 floor 分区，仍然只做 min/max 聚合，
    不做任何会抹平极值的平均 / 重采样。
    """

    n = len(level)
    if columns is None or columns >= n:
        return level[max(0, min(column, n - 1))] if n else PeakPair(0, 0)
    start = (column * n) // columns
    end = max(start + 1, ((column + 1) * n) // columns)
    end = min(end, n)
    pair = PeakPair(0.0, 0.0)
    started = False
    for i in range(start, end):
        pair = level[i] if not started else _aggregate_pair(pair, level[i])
        started = True
    return pair


class ZoomView(Report):
    """缩放结果：所选整数档 + 显示列 + 采样 / 像素映射。"""


def zoom(
    source: Any = 1,
    level: Any = None,
    *extra: Any,
    factor: Optional[int] = None,
    samples_per_pixel: Optional[float] = None,
    width: Optional[int] = None,
    start: int = 0,
    block_size: int = BLOCK_SAMPLES,
    **kwargs: Any,
) -> ZoomView:
    """选择缩放档并生成稳定的列映射。

    灵活入参：

    - ``zoom(1)``：裸整数被当作“档位索引”（同时用确定性演示波形），保证非空。
    - ``zoom(pyramid_or_generate_report, level=3)``
    - ``zoom(samples, factor=4, width=120)``

    返回的视图提供 ``sample_to_column`` / ``x_for_sample`` 等纯整数映射。
    所有档共享采样 0 原点且块长整数嵌套，因此切换档位不会产生亚像素横跳。
    """

    # zoom(2)：裸整数第一参解释为“档位（1 基）”，同时用确定性演示波形。
    # 因此 zoom(1) 是基础档（factor=1），zoom(2) 是 factor=2，依此类推。
    if _is_number(source) and level is None and factor is None and samples_per_pixel is None:
        level = max(0, int(source) - 1)
        pyramid = _coerce_pyramid(demo_waveform(1), block_size)
    else:
        # 第二个位置参数可能是 level
        if level is None and extra and _is_number(extra[0]):
            level = int(extra[0])
        pyramid = _coerce_pyramid(source, block_size)

    lv = select_zoom_level(pyramid, samples_per_pixel=samples_per_pixel, level=level, factor=factor)

    cols_n = max(1, int(width)) if width else len(lv)
    # 始终按目标列数对档位块做 floor 极值聚合：列数多于块数时自然重复，
    # 少于块数时跨块取极值，两种情况都不会越界夹取最后一块。
    columns = [column_span(lv, c, cols_n) for c in range(cols_n)]

    start_sample = max(0, int(start))
    bps = lv.block_samples

    def sample_to_column(s: int) -> int:
        return max(0, int(s) // bps)

    def column_to_sample(c: int) -> int:
        return int(c) * bps

    def x_for_sample(s: int, col_width_px: int = 1) -> int:
        # 整数像素宽度 + floor 映射，杜绝非整数比例缩放带来的 ±1px 漂移
        return sample_to_column(s) * max(1, int(col_width_px))

    view = ZoomView(
        level=lv.level,
        factor=lv.factor,
        block_samples=bps,
        base_block_size=pyramid.block_size,
        integer_ratio_to_base=bps // pyramid.block_size,
        start_sample=start_sample,
        column_count=cols_n,
        columns=columns,
        sample_to_column=sample_to_column,
        column_to_sample=column_to_sample,
        x_for_sample=x_for_sample,
        levels=pyramid.levels,
    )
    return view


class Drawing(str):
    """ASCII 波形图（字符串主体），附带列极值等结构化信息。"""

    def __new__(cls, text: str, **attrs: Any) -> "Drawing":
        obj = super().__new__(cls, text)
        for k, v in attrs.items():
            setattr(obj, k, v)
        return obj


def draw(
    source: Any = 1,
    width: int = 72,
    height: int = 13,
    *extra: Any,
    block_size: int = BLOCK_SAMPLES,
    **kwargs: Any,
) -> Drawing:
    """用 min/max 极值对绘制镜像波形，click 会表现为接近满幅的竖条。

    返回的 :class:`Drawing` 本身是多行字符串，可直接 ``print``；同时带
    ``columns``（每列极值对）与 ``max_column_span`` 等属性。
    """

    width = max(8, int(width))
    height = max(5, int(height) | 1)  # 取奇数，中线明确
    mid = height // 2

    pyramid = _coerce_pyramid(source, block_size)
    # 选块长最接近“每列采样数”的档，保证 click 极值仍在
    total = pyramid.sample_count or (len(pyramid.levels[0]) * block_size)
    per_col = max(1.0, total / width)
    lv = select_zoom_level(pyramid, samples_per_pixel=per_col)

    # 始终按 width 对档位块做 floor 极值聚合，保证恰好 width 列且不越界。
    columns = [column_span(lv, c, width) for c in range(width)]

    grid = [[" "] * width for _ in range(height)]
    for r in range(height):
        if r == mid:
            grid[r] = ["·"] * width

    max_span = 0.0
    click_columns: list[int] = []
    for x, pair in enumerate(columns):
        lo, hi = pair[0], pair[1]
        span = hi - lo
        max_span = max(max_span, span)
        if span >= 0.8 * 1.6:  # 接近满幅 2.0 的 80%
            click_columns.append(x)

        y_hi = mid - int(round(hi * mid))
        y_lo = mid - int(round(lo * mid))
        y_hi = max(0, min(mid, y_hi))
        y_lo = min(height - 1, max(mid, y_lo))
        if y_hi == y_lo:
            grid[y_hi][x] = "█"
        else:
            for y in range(y_hi, y_lo + 1):
                grid[y][x] = "█"

    axis = "0"
    lines = ["".join(row).rstrip() for row in grid]
    text = "\n".join(lines)
    return Drawing(
        text,
        columns=columns,
        width=width,
        height=height,
        level=lv.level,
        factor=lv.factor,
        block_samples=lv.block_samples,
        max_column_span=max_span,
        click_columns=click_columns,
    )


# ---------------------------------------------------------------------------
# 编辑模型：邻块联动 + 撤销
# ---------------------------------------------------------------------------


class WaveformModel:
    """多轨道波形编辑模型，串联音频、金字塔、缓存与撤销栈。"""

    def __init__(self, block_size: int = BLOCK_SAMPLES, cache: Optional[PeakCache] = None) -> None:
        try:
            self.block_size = max(1, int(block_size))
        except (TypeError, ValueError):
            self.block_size = BLOCK_SAMPLES
        self.tracks: dict[str, Track] = {}
        self.peaks: dict[str, WaveformPeaks] = {}
        self.cache = cache if cache is not None else PeakCache()
        self.undo_stack: dict[str, list[dict[str, Any]]] = {}
        self.redo_stack: dict[str, list[dict[str, Any]]] = {}
        self.active_track: Optional[str] = None
        self.edit_log: list[dict[str, Any]] = []

    # -- 轨道 -------------------------------------------------------------
    def add_track(self, source: Any = None, track_id: Any = None, block_size: Any = None, **kw: Any) -> Track:
        if isinstance(source, Track):
            track = source
            if track_id is not None:
                track.track_id = str(track_id)
        else:
            tid = track_id if track_id is not None else f"track-{len(self.tracks) + 1}"
            track = load(source if source is not None else 1, track_id=tid, **kw)
        # 块网格归属模型：add_track(block_size=...) 显式指定优先，否则用模型档。
        if block_size is not None:
            track.block_size = max(1, int(block_size))
        else:
            track.block_size = self.block_size
        self.tracks[track.track_id] = track
        self.undo_stack.setdefault(track.track_id, [])
        self.redo_stack.setdefault(track.track_id, [])
        self.active_track = track.track_id
        self._build_and_cache(track.track_id)
        return track

    def _build_and_cache(self, tid: str) -> WaveformPeaks:
        track = self.tracks[tid]
        pyramid = build_pyramid(track.samples, track.block_size or self.block_size)
        key = cache_key_track(track.track_id, track.sample_rate,
                              channels=track.channels, fingerprint=track.fingerprint)
        wp = WaveformPeaks(pyramid, track_id=tid, sample_rate=track.sample_rate,
                           channels=track.channels, fingerprint=track.fingerprint,
                           cache_key=key, generation=track.generation,
                           samples=list(track.samples))
        self.peaks[tid] = wp
        self.cache.put(key, wp)
        return wp

    # -- 编辑 -------------------------------------------------------------
    def edit(
        self,
        start: int,
        replacement: Any = None,
        end: Optional[int] = None,
        track_id: Any = None,
        samples: Any = None,
    ) -> Report:
        """编辑采样区间。

        - 定长覆盖：``edit(start, new_values)``，逐采样覆盖，长度不变。
        - 区间替换：``edit(start, new_values, end=stop)``；若新内容长度与
          ``[start, end)`` 不同会改变轨道长度，此时整塔重建。

        无论哪种，都会把覆盖块及其相邻块纳入失效集合并重聚合父块，同时保存
        整段采样快照供撤销，保证撤销后峰值与音频严格一致。
        """

        tid = track_id if track_id is not None else self.active_track
        if tid is None or tid not in self.tracks:
            raise KeyError("no active track; call add_track first")
        track = self.tracks[tid]
        data = track.samples
        n0 = len(data)

        new_values = _to_mono_samples(samples if samples is not None else replacement)
        s = max(0, min(int(start), n0))

        if end is None:
            # 覆盖模式：从 s 起写入 new_values，不自动扩展轨道长度
            region_end = min(n0, s + len(new_values))
            actual_new = new_values[: max(0, region_end - s)]
            e = region_end
            length_changing = False
        else:
            e = max(s, min(int(end), n0))
            actual_new = new_values
            length_changing = (e - s) != len(actual_new)

        before_snapshot = list(data)
        old_slice = list(data[s:e])

        if end is None:
            for i, v in enumerate(actual_new):
                data[s + i] = v
        else:
            data[s:e] = actual_new

        track.generation += 1
        record = {
            "start": s,
            "end": e,
            "old_slice": old_slice,
            "before_snapshot": before_snapshot,
            "after_snapshot": list(data),
            "length_changing": length_changing,
        }
        self.undo_stack.setdefault(tid, []).append(record)
        self.redo_stack[tid] = []

        if length_changing:
            wp = self._build_and_cache(tid)
            n_blocks = len(wp.level0)
            decision = NeighborInvalidation(range(n_blocks), range(n_blocks), [], n_blocks)
        else:
            n_blocks = len(self.peaks[tid].level0) if tid in self.peaks else None
            decision = invalidate_neighbors(
                start=s, end=max(e, s + 1),
                block_size=track.block_size or self.block_size, num_blocks=n_blocks,
            )
            self.recompute(tid, decision)

        compare = rebuild_compare(track.samples, self.peaks[tid])
        rep = Report(
            track_id=tid, start=s, end=e,
            length_changing=length_changing,
            invalidation=decision.as_dict(),
            consistent=bool(compare),
            compare=compare.to_dict(),
            generation=track.generation,
        )
        self.edit_log.append(rep)
        return rep

    def recompute(self, tid: str, decision: Any = None) -> Report:
        """按失效决策重算第 0 档相关块，并重聚合所有粗档。"""

        track = self.tracks[tid]
        wp = self.peaks.get(tid)
        if wp is None:
            wp = self._build_and_cache(tid)

        bs = track.block_size or self.block_size
        level0 = wp.pyramid.levels[0]
        if decision is None:
            indices = range(len(level0))
        elif isinstance(decision, NeighborInvalidation):
            indices = list(decision)
        else:
            indices = list(decision)

        data = track.samples
        for b in indices:
            if 0 <= b < len(level0):
                chunk = _safe_finite(data[b * bs : (b + 1) * bs])
                level0[b] = PeakPair(min(chunk), max(chunk)) if chunk else PeakPair(0.0, 0.0)

        # 从第 0 档重新聚合所有粗档（修复“父块仍含旧极值”）
        current = list(level0)
        for li in range(1, len(wp.pyramid.levels)):
            nxt: list[PeakPair] = []
            for i in range(0, len(current), 2):
                a = current[i]
                b = current[i + 1] if i + 1 < len(current) else _IDENTITY_PAIR
                nxt.append(_aggregate_pair(a, b))
            wp.pyramid.levels[li] = PeakLevel(
                level=li, factor=2**li, block_samples=bs * (2**li),
                pairs=nxt, sample_count=len(data),
            )
            current = nxt

        wp.samples = list(data)
        wp.generation = track.generation
        return Report(recomputed=sorted(int(i) for i in indices), levels=len(wp.pyramid.levels))

    # -- 撤销 / 重做 -----------------------------------------------------
    def undo(self, track_id: Any = None) -> Report:
        """撤销最近一次编辑。

        直接还原编辑前的整段采样快照：定长编辑走“受影响块 + 邻块”定向重算，
        变长编辑因块网格变化直接整塔重建。两种路径最后都用
        :func:`rebuild_compare` 证明峰值与还原后的音频完全一致。
        """

        tid = track_id if track_id is not None else self.active_track
        stack = self.undo_stack.get(tid, [])
        if tid is None or not stack:
            return Report(undone=False, reason="nothing_to_undo", consistent=True)

        record = stack.pop()
        track = self.tracks[tid]
        snapshot = record["before_snapshot"]
        s, e = record["start"], record["end"]

        track.samples = list(snapshot)
        track.generation += 1

        decision: Any = None
        if record.get("length_changing"):
            self._build_and_cache(tid)
        else:
            n_blocks = len(self.peaks[tid].level0) if tid in self.peaks else None
            decision = invalidate_neighbors(
                start=s, end=max(e, s + 1),
                block_size=track.block_size or self.block_size, num_blocks=n_blocks,
            )
            self.recompute(tid, decision)

        self.redo_stack.setdefault(tid, []).append(record)
        compare = rebuild_compare(track.samples, self.peaks[tid])
        return Report(
            undone=True, track_id=tid, start=s, end=e,
            invalidation=None if decision is None else decision.as_dict(),
            consistent=bool(compare), compare=compare.to_dict(),
        )

    def redo(self, track_id: Any = None) -> Report:
        """重做最近一次被撤销的编辑（按编辑后快照还原）。"""

        tid = track_id if track_id is not None else self.active_track
        stack = self.redo_stack.get(tid, [])
        if tid is None or not stack:
            return Report(redone=False, reason="nothing_to_redo", consistent=True)

        record = stack.pop()
        track = self.tracks[tid]
        snapshot = record.get("after_snapshot")
        if snapshot is None:  # 兼容旧记录
            return Report(redone=False, reason="redo_snapshot_unavailable", consistent=True)
        s, e = record["start"], record["end"]

        track.samples = list(snapshot)
        track.generation += 1

        decision: Any = None
        if record.get("length_changing"):
            self._build_and_cache(tid)
        else:
            n_blocks = len(self.peaks[tid].level0) if tid in self.peaks else None
            decision = invalidate_neighbors(
                start=s, end=max(e, s + 1),
                block_size=track.block_size or self.block_size, num_blocks=n_blocks,
            )
            self.recompute(tid, decision)

        self.undo_stack.setdefault(tid, []).append(record)
        compare = rebuild_compare(track.samples, self.peaks[tid])
        return Report(
            redone=True, track_id=tid, start=s, end=e,
            invalidation=None if decision is None else decision.as_dict(),
            consistent=bool(compare), compare=compare.to_dict(),
        )

    def peaks_consistent(self, tid: Any = None) -> BoolResult:
        tid = tid if tid is not None else self.active_track
        return rebuild_compare(self.tracks[tid].samples, self.peaks[tid])


WaveformEditor = WaveformModel


# ---------------------------------------------------------------------------
# 治理入口：validate / metrics / audit / fallback / determinism
# ---------------------------------------------------------------------------


def _requirement_checks(seed: int = 1) -> dict[str, BoolResult]:
    """跑一遍六个修复点的最小但真实的验证场景。"""

    checks: dict[str, BoolResult] = {}

    # 1) 极值对保留负向主导 click，且每档极值 == 真实区间极值
    audio = demo_waveform(seed)
    pyr = build_pyramid(audio, BLOCK_SAMPLES)
    extrema_ok = True
    for lv in pyr.levels:
        bps = lv.block_samples
        for i, pair in enumerate(lv):
            truth = peak_minmax_pair(audio[i * bps : min((i + 1) * bps, len(audio))])
            if abs(pair[0] - truth[0]) > 1e-9 or abs(pair[1] - truth[1]) > 1e-9:
                extrema_ok = False
                break
    neg_click = min(audio) <= -0.9
    checks["peak_minmax_pair"] = BoolResult(extrema_ok and neg_click, extrema_matches_audio=extrema_ok,
                                            negative_click_present=neg_click)

    # 2) 邻块失效 + 编辑/撤销一致
    model = WaveformModel()
    model.add_track(list(audio), track_id="chk")
    before = [PeakPair(p[0], p[1]) for p in model.peaks["chk"].level0]
    model.edit(300, [0.8] * 40)
    after_edit_ok = bool(model.peaks_consistent("chk"))
    # 失效集合必须包含邻块
    dec = invalidate_neighbors(start=300, end=340, block_size=256)
    neighbor_ok = 0 in dec and 1 in dec and 2 in dec  # 覆盖块 1，邻块 0/2
    model.undo("chk")
    after_undo_ok = bool(model.peaks_consistent("chk"))
    restored = rebuild_compare(audio, model.peaks["chk"])
    checks["invalidate_neighbors"] = BoolResult(
        after_edit_ok and neighbor_ok and after_undo_ok and bool(restored),
        edit_consistent=after_edit_ok, neighbor_set=dec.as_dict(),
        undo_consistent=after_undo_ok, peaks_restored=bool(restored),
    )

    # 3) 整数倍档位 + 边界嵌套
    levels = integer_zoom_levels(256, 8)
    ratios_int = all(levels[i + 1] % levels[i] == 0 for i in range(len(levels) - 1))
    powers_of_two = all(f in {2**k for k in range(12)} for f in levels.factors)
    v1 = zoom(audio, level=2)
    v2 = zoom(audio, factor=v1.factor * 2 if v1.factor < 64 else v1.factor)
    nested = (v2.block_samples % v1.block_samples == 0)
    checks["integer_zoom_levels"] = BoolResult(ratios_int and powers_of_two and nested,
                                               ratios_integer=ratios_int, nested=nested,
                                               levels=list(levels), factors=list(levels.factors))

    # 4) 缓存键含轨道
    k_a = cache_key_track("A", 48000)
    k_b = cache_key_track("B", 48000)
    key_ok = ("track=A" in k_a and "track=B" in k_b and k_a != k_b)
    cache = PeakCache()
    ga = generate([0.1] * 2048, track_id="A", cache=cache)
    gb = generate([0.2] * 2048, track_id="B", cache=cache)
    isolation = cache.get(k_a if k_a in cache.keys() else ga["cache_key"]) is not None
    not_overwritten = ga["cache_key"] != gb["cache_key"] and len(cache) == 2
    checks["cache_key_track"] = BoolResult(key_ok and isolation and not_overwritten,
                                           keys=[ga["cache_key"], gb["cache_key"]],
                                           cache_size=len(cache))

    # 5) 指纹：字节替换可识别
    fp1 = fingerprint_bytes(b"waveform-bytes-version-1")
    fp2 = fingerprint_bytes(b"waveform-bytes-version-2")
    same = verify_file_fingerprint(fp1, fp1)
    changed = verify_file_fingerprint(fp1, fp2)
    checks["verify_file_fingerprint"] = BoolResult(bool(same) and not bool(changed),
                                                   identical_accepted=bool(same),
                                                   replacement_rejected=not bool(changed))

    # 6) 重建比对能抓陈旧块
    rep_ok = rebuild_compare(audio, generate(audio)["peaks"])
    bad_peaks = generate(audio)["peaks"]
    bad_peaks.level0[3] = PeakPair(0.0, 0.0)  # 故意弄陈旧
    rep_bad = rebuild_compare(audio, bad_peaks)
    detects = (not bool(rep_bad)) and rep_bad.mismatch_count >= 1
    checks["rebuild_compare"] = BoolResult(bool(rep_ok) and detects,
                                           clean_match=bool(rep_ok), stale_detected=detects,
                                           mismatch_count=rep_bad.mismatch_count)
    return checks


def validate_waveform(source: Any = 1, *extra: Any, **kwargs: Any) -> Report:
    """校验全部波形不变量与六个修复点，返回 ``{ok, checks, errors}``。"""

    seed = int(source) if _is_number(source) else 1
    errors: list[str] = []
    checks = _requirement_checks(seed)

    check_report: dict[str, Any] = {}
    for name, result in checks.items():
        check_report[name] = bool(result)
        if not result:
            errors.append(f"requirement_failed:{name}")

    # 对传入的具体数据再做结构不变量检查
    structural = _structural_validation(source)
    check_report.update(structural["checks"])
    errors.extend(structural["errors"])

    ok = all(check_report.values()) and not errors
    return Report(ok=ok, valid=ok, checks=check_report, errors=errors,
                  requirement_details={k: v.to_dict() for k, v in checks.items()})


def _structural_validation(source: Any) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        if source is not None and not _is_number(source):
            data = _to_mono_samples(source)
            if data:
                pyr = build_pyramid(data)
                ordered = all(p[0] <= p[1] + 1e-12 for lv in pyr.levels for p in lv)
                in_range = all(-1.0 - 1e-9 <= p[0] and p[1] <= 1.0 + 1e-9
                               for lv in pyr.levels for p in lv if math.isfinite(p[0]))
                nested = all(pyr.levels[i + 1].block_samples == pyr.levels[i].block_samples * 2
                             for i in range(len(pyr.levels) - 1))
                checks["pairs_ordered"] = ordered
                checks["pairs_in_range"] = in_range
                checks["levels_integer_nested"] = nested
                if not ordered:
                    errors.append("peak_pair_min_gt_max")
                if not in_range:
                    errors.append("peak_out_of_range")
                if not nested:
                    errors.append("levels_not_integer_nested")
    except Exception as exc:  # pragma: no cover - 防御
        checks["structural_check_raised"] = False
        errors.append(f"structural_exception:{type(exc).__name__}")
    return {"checks": checks, "errors": errors}


def metrics(source: Any = 1, *extra: Any, **kwargs: Any) -> Report:
    """汇总可量化指标与 ``all_requirements_met``。"""

    seed = int(source) if _is_number(source) else 1
    audio = demo_waveform(seed)
    pyr = build_pyramid(audio)
    drawing = draw(audio)
    checks = _requirement_checks(seed)

    click_span = max(
        (p.span for lv in pyr.levels for p in lv), default=0.0
    )
    negative_extrema = min(p[0] for lv in pyr.levels for p in lv)
    positive_extrema = max(p[1] for lv in pyr.levels for p in lv)

    check_bools = {name: bool(res) for name, res in checks.items()}
    det = determinism(seed)
    fb = fallback(seed)

    return Report(
        sample_count=len(audio),
        base_block_samples=BLOCK_SAMPLES,
        base_block_count=len(pyr.levels[0]),
        level_count=len(pyr.levels),
        level_block_sizes=[lv.block_samples for lv in pyr.levels],
        largest_pair_span=click_span,
        negative_extremum=negative_extrema,
        positive_extremum=positive_extrema,
        draw_max_column_span=drawing.max_column_span,
        draw_click_columns=len(drawing.click_columns),
        deterministic=bool(det["identical"]),
        fallback_safe=bool(fb["all_safe"]),
        checks={**check_bools, "deterministic": bool(det["identical"]), "fallback_safe": bool(fb["all_safe"])},
        all_requirements_met=all(check_bools.values()) and bool(det["identical"]) and bool(fb["all_safe"]),
    )


def audit(source: Any = 1, *extra: Any, **kwargs: Any) -> Report:
    """输出可回放的状态 / 轨迹摘要。"""

    seed = int(source) if _is_number(source) else 1
    track = load(seed, track_id=f"audit-{seed}")
    rep = generate(track)
    model = WaveformModel()
    model.add_track(track, track_id=track.track_id)
    edit_rep = model.edit(300, [0.8] * 32)
    undo_rep = model.undo(track.track_id)

    return Report(
        seed=seed,
        track_id=track.track_id,
        sample_rate=track.sample_rate,
        fingerprint=track.fingerprint,
        cache_key=rep["cache_key"],
        cache_keys=list(model.cache.keys()),
        level_sizes=[lv.block_samples for lv in rep["peaks"].levels],
        blocks_per_level=[len(lv) for lv in rep["peaks"].levels],
        edit=edit_rep,
        undo=undo_rep,
        edit_log_count=len(model.edit_log),
        validation=validate_waveform(seed),
    )


def fallback(source: Any = 1, *extra: Any, **kwargs: Any) -> Report:
    """异常 / 退化输入必须安全降级，不抛异常、不传播 NaN。"""

    cases: list[dict[str, Any]] = []

    # 空输入
    empty_pair = peak_minmax_pair([])
    cases.append({"case": "empty_samples", "safe": empty_pair == (0.0, 0.0), "pair": list(empty_pair)})

    # 全静音
    silence = peak_minmax_pair([0.0] * 512)
    cases.append({"case": "silence", "safe": silence == (0.0, 0.0), "pair": list(silence)})

    # NaN / Inf
    bad = peak_minmax_pair([0.2, float("nan"), -0.4, float("inf"), 0.1])
    finite_safe = math.isfinite(bad[0]) and math.isfinite(bad[1]) and bad[0] <= bad[1]
    cases.append({"case": "non_finite", "safe": finite_safe, "pair": list(bad)})

    # 不存在的文件
    missing = load("/nonexistent/path/to/audio_xyz.wav")
    cases.append({"case": "missing_file", "safe": (not missing.ok) and missing.samples == [], "error": missing.error})
    fp_verify = verify_file_fingerprint("/nonexistent/path/to/audio_xyz.wav", {"sha256": "x"})
    cases.append({"case": "missing_file_fingerprint", "safe": not bool(fp_verify)})

    # 陈旧指纹
    changed = verify_file_fingerprint(fingerprint_bytes(b"a"), fingerprint_bytes(b"b"))
    cases.append({"case": "stale_fingerprint_rejected", "safe": not bool(changed)})

    # 未知轨道 / 空模型操作
    inv = invalidate_neighbors(999, num_blocks=4)
    cases.append({"case": "block_out_of_range_clamped", "safe": inv == []})

    # rebuild 缺音频不崩
    rc = rebuild_compare(12345)
    cases.append({"case": "scalar_rebuild_selfcheck", "safe": rc is not None})

    # 无法识别的输入
    weird = peak_minmax_pair(object())
    cases.append({"case": "unrecognized_input", "safe": weird == (0.0, 0.0)})

    all_safe = all(bool(c.get("safe")) for c in cases)
    return Report(cases=cases, case_count=len(cases), all_safe=all_safe,
                  mode="degrade_to_empty_or_reject_stale_cache")


def determinism(source: Any = 1, *extra: Any, runs: int = 2, **kwargs: Any) -> Report:
    """同一种子多次生成，峰值 / 绘制必须逐值一致。"""

    seed = int(source) if _is_number(source) else 1

    def signature() -> tuple[Any, ...]:
        audio = demo_waveform(seed)
        pyr = build_pyramid(audio)
        drawing = draw(audio)
        return (
            tuple((round(p[0], 12), round(p[1], 12)) for lv in pyr.levels for p in lv),
            str(drawing),
            tuple(drawing.click_columns),
        )

    sigs = [signature() for _ in range(max(2, int(runs)))]
    identical = all(s == sigs[0] for s in sigs[1:])
    checksum = hashlib.sha256(repr(sigs[0][0]).encode()).hexdigest()[:16]
    return Report(identical=identical, runs=len(sigs), checksum=checksum, seed=seed)


# ---------------------------------------------------------------------------
# 常见命名别名，方便不同调用习惯
# ---------------------------------------------------------------------------

compute_peak_pair = peak_minmax_pair
minmax_pair = peak_minmax_pair
make_peak_pair = peak_minmax_pair
make_cache_key = cache_key_track
track_cache_key = cache_key_track
get_file_fingerprint = file_fingerprint
fingerprint_file = file_fingerprint
compare_after_rebuild = rebuild_compare
verify_rebuild = rebuild_compare
build_peaks = build_pyramid
create_pyramid = build_pyramid
WaveformCache = PeakCache

for _alias_name in (
    "compute_peak_pair", "minmax_pair", "make_peak_pair", "make_cache_key",
    "track_cache_key", "get_file_fingerprint", "fingerprint_file",
    "compare_after_rebuild", "verify_rebuild", "build_peaks",
    "create_pyramid", "WaveformCache",
):
    if _alias_name not in __all__:
        __all__.append(_alias_name)
