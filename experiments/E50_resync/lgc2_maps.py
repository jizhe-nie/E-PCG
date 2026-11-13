#!/usr/bin/env python
"""E50 phase 2 -- per-recording shift maps for LGC2-Net, from E50's own offset estimates.

    python experiments/E50_resync/lgc2_maps.py
    bash experiments/E50_resync/run_lgc2.sh

LGC2-Net runs at 2 kHz, so every map entry is (offset_ms - correction_ms) * 2 samples.
The offsets and the delay operator's estimates are exactly those of the PINet arm
(e50_oof.npz), so both classifiers face identical desynchronisation and identical
corrections. Raw +-80 ms conditions are already on disk from E46 and are not rerun.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
sys.path.insert(0, str(OUT))
from run import DEADBAND_MS, SITES, E49  # noqa: E402

z = np.load(OUT / "e50_oof.npz")
subs = [str(s) for s in E49.E44._E12.build_cache(False)["subjects"]]
db = lambda d: np.sign(d) * np.maximum(np.abs(d) - DEADBAND_MS, 0.0)
MAPS = {"random_raw": z["off__random"],
        "random_deadband": z["off__random"] - np.round(db(z["dhat__random"])),
        "fixed+80_deadband": z["off__fixed+80"] - np.round(db(z["dhat__fixed+80"])),
        "fixed-80_deadband": z["off__fixed-80"] - np.round(db(z["dhat__fixed-80"])),
        "none_deadband": -np.round(db(z["dhat__none"]))}
(OUT / "lgc2_maps").mkdir(exist_ok=True)
for name, M in MAPS.items():
    m = {f"{s}_{site}": int(2 * M[i, j]) for i, s in enumerate(subs)
         for j, site in enumerate(SITES) if int(M[i, j])}
    (OUT / "lgc2_maps" / f"{name}.json").write_text(json.dumps(m), encoding="utf-8")
    print(f"{name:20s} {len(m)} non-zero entries, |shift| mean "
          f"{np.mean(np.abs(list(m.values()))) / 2 if m else 0:.1f} ms")
