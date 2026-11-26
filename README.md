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

## 修复点与实现对照

| 缺陷 | 修复 | 关键入口 |
| --- | --- | --- |
| 最大缩放档只剩每档最大值，负向 / 双极高频 click 被吞 | 每个峰值块保留 `(min, max)` 极值对；层级聚合只做 `min(低点)` / `max(高点)`，极值满足结合律，任意档极值都等于该采样区间的真实极值 | `peak_minmax_pair` / `PeakPair` / `build_pyramid` |
| 编辑后只更新被改块，邻块与粗档父块仍含旧极值，撤销后峰值与音频不一致 | 编辑区间换算成基础块后，把被改块及其 ±1 邻块一并失效重算，再从第 0 档重新聚合所有父块；保存整段采样快照，撤销走同一套重算，并用重建比对证明一致 | `invalidate_neighbors` / `WaveformModel.edit` / `undo` / `recompute` |
| 中间档按最近非整数比例缩放，波形左右跳一像素 | 档位固定为 `256 × 2^k`（`[256,512,1024,…]`，倍率 `[1,2,4,…]`），相邻档恒为整数倍、边界互相嵌套；列映射全部用整数 floor，共享采样 0 原点 | `integer_zoom_levels` / `select_zoom_level` / `zoom` |
| 多轨共用一份峰值缓存，键只含采样率，等长轨道互相覆盖 | 缓存键显式包含 `track=<id>`（以及采样率 / 声道 / 可选指纹），两条等长同采样率轨道得到不同键，存取互不覆盖 | `cache_key_track` / `PeakCache` |
| 缓存按路径绑定，同路径文件被替换后仍是旧波形 | 记录 `size + mtime_ns + sha256` 指纹；即使替换后大小相同，sha256 变化也会判定失效 | `file_fingerprint` / `verify_file_fingerprint` |
| 重建后不知道缓存是否仍陈旧 | 从当前音频重建并逐档逐块比对，返回是否一致及每个失配块的档号 / 块号 / 期望值 / 实际值；可直接比对模型、`WaveformPeaks` 或两份峰值 | `rebuild_compare` |

流水线由 `load`（list/dict/种子 / 真实 PCM WAV）、`generate`（构造金字塔并写轨道缓存，
`BackgroundPeakGenerator` 用 generation 代际丢弃过期后台结果）、`zoom`（整数档视图与
采样↔列映射）、`draw`（按 min/max 极值对绘制 ASCII 镜像波形，click 呈近满幅竖条）组成；
`validate_waveform` / `metrics` / `audit` 提供不变量校验、`all_requirements_met` 指标与
可回放编辑 / 撤销轨迹，`fallback` 覆盖空输入、NaN/Inf、文件缺失、陈旧指纹与越界块，
`determinism` 双跑逐值比对保证可复现。

## 验证

```bash
python -m pytest -q
```

除原有 15 个契约冒烟外，`tests/test_waveform_fixes.py` 用 50 个行为用例固化上述六个修复点，
包括：每档极值与真实区间一致、click 在最粗档仍存在、跨块编辑的邻块失效、编辑 / 撤销 / 重做后
峰值逐块还原、变长编辑撤销、整数档嵌套与零锚点、两等长轨道缓存隔离、同路径等长替换的指纹识别、
陈旧第 0 档块与陈旧粗档父块的定位、Report / 双峰值 / 立体声帧等多种入参形态、NaN/Inf 与越界
输入、真实 WAV 往返、后台生成落缓存与过期结果丢弃等。全部 65 个用例通过；另用 1500 组随机
长度 / 块档 / 编辑 / 撤销重做迭代验证不变量。

