"""波形峰值缓存六个修复点的行为回归测试。

这些测试把题面描述的故障逐个固化成可执行断言：

1. 峰值保留 min/max 极值对，负向 click 在最粗档也不丢；
2. 编辑联动邻块与粗档父块，撤销后峰值与音频严格一致；
3. 缩放档位是基础块的整数倍（2 的幂），边界嵌套、映射为整数；
4. 缓存键含轨道，等长同采样率轨道互不覆盖；
5. 文件指纹识别同路径替换；
6. 重建后比对能抓出陈旧块 / 陈旧父块。

外加 load/generate/zoom/draw 流水线与 validate/metrics/audit/fallback/
determinism 治理入口，以及真实 WAV 往返与后台生成。
"""

import math
import struct
import wave

import pytest

import media.waveform_peak_cache as wf
from media.waveform_peak_cache import (
    BLOCK_SAMPLES,
    BackgroundPeakGenerator,
    PeakCache,
    PeakPair,
    WaveformModel,
    build_pyramid,
    cache_key_track,
    demo_waveform,
    draw,
    file_fingerprint,
    generate,
    integer_zoom_levels,
    invalidate_neighbors,
    load,
    peak_minmax_pair,
    rebuild_compare,
    verify_file_fingerprint,
    zoom,
)


# ---------------------------------------------------------------------------
# 修复点 1：极值对
# ---------------------------------------------------------------------------


class TestMinMaxPair:
    def test_basic_pair_order_and_values(self):
        assert tuple(peak_minmax_pair([0.0, 0.5, -0.9, 0.2])) == pytest.approx((-0.9, 0.5))

    def test_unpack_and_attrs(self):
        p = peak_minmax_pair([-0.2, 0.7])
        lo, hi = p
        assert lo == -0.2 and hi == 0.7
        assert p.min == -0.2 and p.max == 0.7
        assert p.span == pytest.approx(0.9)
        # 与普通二元组相等
        assert p == (-0.2, 0.7)
        assert isinstance(p, PeakPair)

    def test_empty_and_silence_are_zero_pair(self):
        assert peak_minmax_pair([]) == (0.0, 0.0)
        assert peak_minmax_pair([0.0] * 32) == (0.0, 0.0)

    def test_non_finite_is_ignored_safely(self):
        p = peak_minmax_pair([0.2, float("nan"), -0.4, float("inf")])
        assert math.isfinite(p.min) and math.isfinite(p.max)
        assert p == pytest.approx((-0.4, 0.2))

    def test_negative_dominant_click_invisible_to_max_only(self):
        # 负向占主导的 click：只有 -0.95 与很小的 +0.10
        block = [0.0] * 100 + [-0.95, 0.10] + [0.0] * 154
        pair = peak_minmax_pair(block)
        # 极值对保留了深负端与完整跨度 -> click 可见
        assert pair.min <= -0.9
        assert pair.span >= 1.0
        # 复现缺陷：只存有符号 max 时，峰值只有 0.10，click 被吞
        max_only = max(block)
        assert max_only < 0.2

    def test_extrema_match_truth_at_every_level(self):
        audio = demo_waveform(3)
        pyr = build_pyramid(audio, BLOCK_SAMPLES)
        for lv in pyr.levels:
            bps = lv.block_samples
            for i, pair in enumerate(lv):
                truth = peak_minmax_pair(audio[i * bps : (i + 1) * bps])
                assert pair.min == pytest.approx(truth.min, abs=1e-9)
                assert pair.max == pytest.approx(truth.max, abs=1e-9)

    def test_click_survives_max_zoom_out(self):
        audio = demo_waveform(1)
        pyr = build_pyramid(audio)
        top = pyr.levels[-1][0]  # 最粗档只剩一个块
        # 结合律保证最粗档极值就是整段音频的真实极值
        assert top.min == pytest.approx(min(audio))
        assert top.max == pytest.approx(max(audio))
        assert top.min <= -0.9  # 负向 click 在最大缩放档仍在

    def test_blocking_entry_returns_level(self):
        audio = [0.1, -0.1, 0.5, -0.5] * 128  # 512 samples
        lv = peak_minmax_pair(audio, block_size=256)
        assert len(lv) == 2
        assert lv.block_samples == 256
        assert lv[0] == pytest.approx((-0.5, 0.5))


# ---------------------------------------------------------------------------
# 修复点 2：编辑联动邻块 + 撤销
# ---------------------------------------------------------------------------


class TestNeighborInvalidation:
    def test_scalar_includes_self_and_neighbors(self):
        dec = invalidate_neighbors(3)
        assert dec == [2, 3, 4]
        assert dec.changed == [3]
        assert dec.neighbors == [2, 4]

    def test_boundary_clamped_by_num_blocks(self):
        assert invalidate_neighbors(0, num_blocks=5) == [0, 1]
        assert invalidate_neighbors(4, num_blocks=5) == [3, 4]
        assert invalidate_neighbors(99, num_blocks=4) == []

    def test_multiple_changed_union(self):
        dec = invalidate_neighbors([3, 10], num_blocks=12)
        assert dec == [2, 3, 4, 9, 10, 11]

    def test_sample_range_maps_to_blocks_and_neighbors(self):
        dec = invalidate_neighbors(start=300, end=340, block_size=256)
        # 300..340 落在块 1；邻块 0、2 一起失效
        assert dec.changed == [1]
        assert dec == [0, 1, 2]
        assert dec.neighbors == [0, 2]


class TestEditUndoConsistency:
    def _model(self):
        model = WaveformModel(block_size=256)
        model.add_track(demo_waveform(7), track_id="vox")
        return model

    def test_edit_near_boundary_invalidates_neighbors(self):
        model = self._model()
        rep = model.edit(250, [0.8] * 20, track_id="vox")  # 跨块 0/1
        assert rep["consistent"] is True
        inv = rep["invalidation"]["invalidate"]
        assert 0 in inv and 1 in inv and 2 in inv
        assert rebuild_compare(model.tracks["vox"].samples, model.peaks["vox"])

    def test_edit_then_undo_restores_samples_and_peaks_exactly(self):
        model = self._model()
        tid = "vox"
        original_samples = list(model.tracks[tid].samples)
        original_pairs = [tuple(p) for p in model.peaks[tid].level0]

        model.edit(300, [0.7] * 48, track_id=tid)
        assert model.peaks_consistent(tid)

        undone = model.undo(tid)
        assert undone["undone"] is True
        assert undone["consistent"] is True
        assert model.tracks[tid].samples == original_samples
        assert [tuple(p) for p in model.peaks[tid].level0] == original_pairs

    def test_length_changing_edit_and_undo(self):
        model = self._model()
        tid = "vox"
        original_len = len(model.tracks[tid].samples)
        original_samples = list(model.tracks[tid].samples)

        # 用 end 做变长替换：40 采样 -> 5 采样
        model.edit(100, [0.2, 0.3, 0.4, 0.5, 0.6], end=140, track_id=tid)
        assert len(model.tracks[tid].samples) == original_len - 35
        assert model.peaks_consistent(tid)

        model.undo(tid)
        assert len(model.tracks[tid].samples) == original_len
        assert model.tracks[tid].samples == original_samples
        assert model.peaks_consistent(tid)

    def test_stale_neighbor_or_parent_is_detected(self):
        model = self._model()
        tid = "vox"
        assert model.peaks_consistent(tid)
        # 模拟“只改了被改块，邻块仍旧值”造成的陈旧峰值
        model.peaks[tid].level0[4] = PeakPair(0.0, 0.0)
        bad = model.peaks_consistent(tid)
        assert not bad
        assert bad.mismatch_count >= 1


# ---------------------------------------------------------------------------
# 修复点 3：整数倍档位
# ---------------------------------------------------------------------------


class TestIntegerZoom:
    def test_default_levels_are_base_times_powers_of_two(self):
        levels = integer_zoom_levels(256, 8)
        assert levels[0] == 256
        assert list(levels) == [256 * (2**i) for i in range(8)]
        assert levels.factors == [1, 2, 4, 8, 16, 32, 64, 128]

    def test_every_adjacent_ratio_is_integer(self):
        levels = integer_zoom_levels()
        for a, b in zip(levels, levels[1:]):
            assert b % a == 0
            assert b // a >= 2

    def test_contract_scalar(self):
        levels = integer_zoom_levels(1)
        assert len(levels) >= 1
        assert all(isinstance(x, int) and x >= 1 for x in levels)

    def test_coarse_columns_are_integer_aggregation_of_fine(self):
        audio = demo_waveform(2)
        pyr = build_pyramid(audio)
        l0, l1, l2 = pyr.levels[0], pyr.levels[1], pyr.levels[2]
        for j, pair in enumerate(l1):
            children = [l0[2 * j], l0[2 * j + 1]]
            assert pair.min == pytest.approx(min(c.min for c in children))
            assert pair.max == pytest.approx(max(c.max for c in children))
        for j, pair in enumerate(l2):
            assert pair.min == pytest.approx(min(l0[4 * j + k].min for k in range(4)))
            assert pair.max == pytest.approx(max(l0[4 * j + k].max for k in range(4)))

    def test_zoom_views_integer_nested_and_zero_anchor(self):
        audio = demo_waveform(1)
        views = [zoom(audio, factor=f) for f in (1, 2, 4, 8)]
        blocks = [v.block_samples for v in views]
        assert blocks == [256, 512, 1024, 2048]
        for finer, coarser in zip(views, views[1:]):
            assert coarser.block_samples % finer.block_samples == 0
        # 全局原点在每一档都映射到像素 0，不产生亚像素左偏移
        for v in views:
            assert v.x_for_sample(0) == 0
            assert v.sample_to_column(0) == 0
        # 落在高倍边界上的采样在各档都是整数列边界
        s = 256 * 8  # 2048
        assert [v.sample_to_column(s) for v in views] == [8, 4, 2, 1]

    def test_zoom_contract_scalar_is_first_gear(self):
        v = zoom(1)
        assert v.factor == 1
        assert v.block_samples == BLOCK_SAMPLES
        assert len(v.columns) > 0


# ---------------------------------------------------------------------------
# 修复点 4：缓存键含轨道
# ---------------------------------------------------------------------------


class TestTrackCacheKey:
    def test_key_contains_track_and_sample_rate(self):
        k = cache_key_track("guitar", 44100)
        assert "track=guitar" in k
        assert "sr=44100" in k

    def test_equal_length_tracks_get_distinct_keys(self):
        k_a = cache_key_track(7, 22050)
        k_b = cache_key_track(8, 22050)
        assert k_a != k_b
        # 复现缺陷：只含采样率的旧键会相撞
        assert f"sr={22050}" == f"sr={22050}"

    def test_cache_does_not_overwrite_equal_length_tracks(self):
        cache = PeakCache()
        rep_a = generate([0.5] * 2048, track_id="A", sample_rate=48000, cache=cache)
        rep_b = generate([-0.5] * 2048, track_id="B", sample_rate=48000, cache=cache)
        assert rep_a["cache_key"] != rep_b["cache_key"]
        assert len(cache) == 2

        pa = cache.get(rep_a["cache_key"])
        pb = cache.get(rep_b["cache_key"])
        # 取回来的必须是各自的内容，而不是被另一条等长轨道覆盖后的结果
        assert pa.level0[0].max == pytest.approx(0.5)
        assert pb.level0[0].min == pytest.approx(-0.5)

    def test_invalidate_track_removes_all_its_keys(self):
        cache = PeakCache()
        generate([0.1] * 512, track_id="A", cache=cache)
        generate([0.2] * 512, track_id="A", fingerprint={"sha256": "deadbeef"}, cache=cache)
        generate([0.3] * 512, track_id="B", cache=cache)
        removed = cache.invalidate_track("A")
        assert removed == 2
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# 修复点 5：文件指纹
# ---------------------------------------------------------------------------


class TestFileFingerprint:
    def test_roundtrip_and_same_path_replacement_detected(self, tmp_path):
        f = tmp_path / "take.wav"
        f.write_bytes(b"RIFF----WAVE original content " * 4)
        fp1 = file_fingerprint(f)
        assert fp1["exists"] is True
        assert verify_file_fingerprint(f, fp1)

        # 同路径整体替换，且刻意保持相同长度，sha256 仍必须识别出来
        f.write_bytes(b"RIFF----WAVE replaced content " * 4)
        fp2 = file_fingerprint(f)
        assert fp1["sha256"] != fp2["sha256"]
        assert not verify_file_fingerprint(f, fp1)
        assert verify_file_fingerprint(f, fp2)

    def test_fingerprint_pairwise_compare(self):
        assert verify_file_fingerprint({"sha256": "aaa"}, {"sha256": "aaa"})
        assert not verify_file_fingerprint({"sha256": "aaa"}, {"sha256": "bbb"})

    def test_missing_file_is_safe_false(self):
        res = verify_file_fingerprint("/no/such/file_xyz.wav", {"sha256": "x"})
        assert not res
        assert res.reason == "file_missing"

    def test_load_real_wav_and_verify(self, tmp_path):
        path = tmp_path / "click.wav"
        sr = 22050
        n = sr // 10
        frames = []
        for i in range(n):
            v = 0.3 * math.sin(2 * math.pi * 440 * i / sr)
            if i == 1000:
                v = -0.95
            frames.append(int(max(-1, min(1, v)) * 32767))
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(struct.pack(f"<{n}h", *frames))

        track = load(str(path), track_id="click")
        assert track.ok
        assert track.sample_rate == sr
        assert len(track.samples) == n
        assert min(track.samples) <= -0.9
        assert verify_file_fingerprint(path, track.fingerprint)


# ---------------------------------------------------------------------------
# 修复点 6：重建后比对
# ---------------------------------------------------------------------------


class TestRebuildCompare:
    def test_clean_peaks_match(self):
        audio = demo_waveform(4)
        peaks = generate(audio)["peaks"]
        res = rebuild_compare(audio, peaks)
        assert res
        assert res.mismatch_count == 0

    def test_stale_level0_block_located(self):
        audio = demo_waveform(4)
        peaks = generate(audio)["peaks"]
        peaks.level0[5] = PeakPair(0.0, 0.0)
        res = rebuild_compare(audio, peaks)
        assert not res
        assert res.mismatch_count >= 1
        assert any(m.get("index") == 5 and m.get("level") == 0 for m in res.mismatches)

    def test_stale_coarse_parent_located(self):
        audio = demo_waveform(4)
        peaks = generate(audio)["peaks"]
        coarse = peaks.levels[2]
        coarse[0] = PeakPair(0.0, 0.0)
        res = rebuild_compare(audio, peaks)
        assert not res
        assert any(m.get("level") == 2 for m in res.mismatches)

    def test_model_self_compare(self):
        model = WaveformModel()
        model.add_track(demo_waveform(5), track_id="t")
        assert rebuild_compare(model)


# ---------------------------------------------------------------------------
# 绘制：click 必须在画面上形成明显竖条
# ---------------------------------------------------------------------------


class TestDraw:
    def test_drawing_shows_click_columns(self):
        audio = demo_waveform(1)
        d = draw(audio, width=72, height=15)
        assert isinstance(str(d), str) and len(str(d)) > 0
        # 至少有一列跨度很大（click），且能定位到 click 列
        assert d.max_column_span >= 1.0
        assert len(d.click_columns) >= 1
        # 负向 click 列保留深负端
        assert any(p.min <= -0.8 for p in d.columns)

    def test_draw_contract_scalar(self):
        d = draw(1)
        assert d is not None
        assert "█" in str(d)


# ---------------------------------------------------------------------------
# 后台生成
# ---------------------------------------------------------------------------


class TestBackground:
    def test_background_generation_eventually_cached(self):
        cache = PeakCache()
        bg = BackgroundPeakGenerator(cache=cache)
        try:
            track = load(demo_waveform(1), track_id="bg")
            fut = bg.submit(track)
            rep = fut.result(timeout=10)
            assert rep["peaks"] is not None
            assert cache.contains(rep["cache_key"])
        finally:
            bg.shutdown()

    def test_stale_background_result_is_discarded(self):
        import threading

        cache = PeakCache()
        bg = BackgroundPeakGenerator(cache=cache, max_workers=1)
        try:
            entered = threading.Event()
            release = threading.Event()

            def gate():
                entered.set()
                release.wait(timeout=5)

            blocker = bg._executor.submit(gate)  # 占住唯一 worker
            assert entered.wait(timeout=5)

            track = load([0.1] * 512, track_id="race")
            fut = bg.submit(track, generation=1)  # 排在 gate 后面
            # 模拟排队期间又发生一次编辑，代际推进到 2
            bg._last_generation["race"] = 2

            release.set()
            blocker.result(timeout=5)
            rep = fut.result(timeout=10)

            assert rep["stale_discarded"] is True
            # 过期结果写入后必须被撤销，不能覆盖新峰值
            assert not cache.contains(rep["cache_key"])
        finally:
            release.set()
            bg.shutdown()


# ---------------------------------------------------------------------------
# 治理入口
# ---------------------------------------------------------------------------


class TestGovernance:
    def test_validate_all_ok(self):
        rep = wf.validate_waveform(1)
        assert rep["ok"] is True, rep.get("errors")
        for name, ok in rep["checks"].items():
            assert ok, name

    def test_metrics_all_requirements_met(self):
        m = wf.metrics(1)
        assert m["all_requirements_met"] is True
        assert m["level_count"] >= 5
        assert m["draw_click_columns"] >= 1

    def test_fallback_all_safe(self):
        fb = wf.fallback(1)
        assert fb["all_safe"] is True
        assert fb["case_count"] >= 7

    def test_determinism_identical(self):
        d = wf.determinism(1)
        assert d["identical"] is True
        assert d["runs"] >= 2

    def test_audit_runs_edit_and_undo_consistent(self):
        a = wf.audit(1)
        assert a["edit"]["consistent"] is True
        assert a["undo"]["consistent"] is True
        assert a["validation"]["ok"] is True


# ---------------------------------------------------------------------------
# 复查发现的边界问题回归
# ---------------------------------------------------------------------------


class TestReviewEdgeCases:
    def test_rebuild_compare_accepts_generate_report(self):
        audio = demo_waveform(1)
        report = generate(audio)  # Report，真正的 WaveformPeaks 在 ["peaks"]
        assert rebuild_compare(audio, report)
        report["peaks"].level0[5] = PeakPair(0.0, 0.0)
        bad = rebuild_compare(audio, report)
        assert not bad
        assert bad.mismatch_count >= 1
        # 标量自检结果也必须带齐明细字段，不能 AttributeError
        scalar = rebuild_compare(1)
        assert scalar.mismatch_count == 0
        assert scalar.mismatches == []

    def test_rebuild_compare_peak_to_peak_plain_tuples(self):
        a = [(0.1, 0.2), (-0.4, 0.5)]
        b = [(0.1, 0.2), (-0.4, 0.5)]
        assert rebuild_compare(a, b)
        c = [(0.1, 0.2), (0.0, 0.5)]
        res = rebuild_compare(a, c)
        assert not res and res.mismatch_count >= 1

    def test_extensionless_existing_file_not_treated_as_fingerprint(self, tmp_path, monkeypatch):
        f = tmp_path / "audiocache"  # 无扩展名
        f.write_bytes(b"waveform peak cache binary")
        fp = file_fingerprint(f)
        monkeypatch.chdir(tmp_path)
        assert verify_file_fingerprint("audiocache", fp)

    def test_stereo_frames_are_downmixed_not_peak_pairs(self):
        frames = [(0.1, 0.2), (-0.9, 0.3), (0.4, -0.4)] * 200
        assert wf._is_peak_structure(frames) is False
        mono = [(l + r) / 2.0 for l, r in frames]
        peaks = generate(mono)["peaks"]
        assert rebuild_compare(frames, peaks)

    def test_invalidate_neighbors_skips_non_finite(self):
        assert invalidate_neighbors([float("nan"), 3, float("inf")]) == [2, 3, 4]
        assert invalidate_neighbors(float("nan")) == []

    def test_model_block_size_is_honored(self):
        model = WaveformModel(block_size=128)
        track = model.add_track([0.1] * 1024, track_id="t")
        assert track.block_size == 128
        assert len(model.peaks["t"].level0) == 8

    def test_redo_reapplies_undone_edit(self):
        model = WaveformModel()
        model.add_track(demo_waveform(2), track_id="t")
        original = list(model.tracks["t"].samples)
        model.edit(300, [0.7] * 32, track_id="t")
        edited = list(model.tracks["t"].samples)
        assert edited != original

        undone = model.undo("t")
        assert undone["consistent"] is True
        assert model.tracks["t"].samples == original

        redone = model.redo("t")
        assert redone["redone"] is True
        assert redone["consistent"] is True
        assert model.tracks["t"].samples == edited

        # 空重做安全返回而非抛异常
        model.redo("t")
        again = model.redo("t")
        assert again["redone"] is False
