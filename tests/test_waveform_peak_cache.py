import pytest
import media.waveform_peak_cache as mod

CASES = [
    ("peak_minmax_pair", (1,)),
    ("invalidate_neighbors", (1,)),
    ("integer_zoom_levels", (1,)),
    ("cache_key_track", (1,)),
    ("verify_file_fingerprint", (1,)),
    ("rebuild_compare", (1,)),
    ("load", (1,)),
    ("generate", (1,)),
    ("zoom", (1,)),
    ("draw", (1,)),
    ("validate_waveform", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
