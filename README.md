# pairwise-waveform-peak-cache

## 领域

游戏、图形与多媒体

## 主题

波形峰值缓存

## 说明

极值对峰值、邻块失效与缓存键含轨道。

初始状态下 `media/waveform_peak_cache.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
