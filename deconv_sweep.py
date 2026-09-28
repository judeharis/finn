# Jude: Created
"""
Sweep several deconv configurations through FINN rtlsim and record, per config,
both the measured `cycles_rtlsim` and the predicted `get_exp_cycles`.

Run inside the FINN container:

    cd $FINN_ROOT && python -m deconv_sweep

Results are appended to deconv_sweep_results.json after EVERY config, so a crash
or a Ctrl-C part way through does not lose the configs that already ran (each one
costs a Vitis HLS synthesis). Re-running skips configs already in that file, so
it resumes rather than repeating work.

This drives the same rtlsim path as test_fpgadataflow_deconv_revd2 but calls the
transformations directly instead of the test function, because the test ends in
`assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)` -- which is exactly the
quantity being measured, so letting it fire would abort the sweep.
"""
import json
import os
import traceback
import warnings

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from tests.fpgadataflow.test_fpgadataflow_deconv import (
    create_deconv_node,
    set_up_reference_model,
    target_clk_ns,
    test_fpga_part,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deconv_sweep_results.json")

# (name, K, S, P, H, W, CI, CO, PE, SIMD)
# All S=2. Every axis the existing S=1 cosim data confounds gets moved here:
# P, H, K, H!=W, and PE/SIMD folding.
CONFIGS = [
    ("baseline",     4, 2, 1, 8, 8, 2, 3, 1, 1),  # repeat of the known 7350 point
    ("vary_P",       4, 2, 2, 8, 8, 2, 3, 1, 1),  # P=2 -> CROP becomes 0
    ("vary_H",       4, 2, 1, 4, 4, 2, 3, 1, 1),
    ("H_neq_W",      4, 2, 1, 4, 8, 2, 3, 1, 1),  # first non-square point anywhere
    ("vary_K_big",   6, 2, 2, 8, 8, 2, 3, 1, 1),  # K=6, KK=3
    ("vary_K_small", 6, 2, 2, 4, 4, 2, 3, 1, 1),
    ("folded",       4, 2, 1, 8, 8, 4, 4, 2, 2),  # CF=2, SF=2
]


def run_one(K, S, P, H, W, CI, CO, PE, SIMD):
    idt = wdt = DataType["INT8"]
    odt = DataType["INT32"]
    idim = [H, W]
    stride = [S, S]

    ref_model, w_tensor = set_up_reference_model(idt, wdt, odt, K, idim, CI, CO, stride, P)
    model = create_deconv_node(idt, wdt, odt, K, idim, CI, CO, stride, P, w_tensor)

    odim_h = (H - 1) * S - 2 * P + (K - 1) + 1
    odim_w = (W - 1) * S - 2 * P + (K - 1) + 1

    input_tensor = gen_finn_dt_tensor(idt, [1, CI, H, W])
    y_expected = oxe.execute_onnx(ref_model, {"inp": input_tensor})["outp"]

    for n in model.graph.node:
        if n.op_type.startswith("Deconvolution_hls"):
            inst = getCustomOp(n)
            inst.set_nodeattr("PE", PE)
            inst.set_nodeattr("SIMD", SIMD)

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    model = model.transform(SetExecMode("rtlsim"))

    y_produced = oxe.execute_onnx(model, {"inp": input_tensor.transpose(0, 2, 3, 1)})["outp"]
    shape_ok = y_produced.shape == (1, odim_h, odim_w, CO)
    values_ok = bool((y_produced.transpose(0, 3, 1, 2) == y_expected).all()) if shape_ok else False

    node = model.get_nodes_by_op_type("Deconvolution_hls")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = int(inst.get_nodeattr("cycles_rtlsim"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the extrapolation warning is expected here
        exp_cycles = int(model.analysis(exp_cycles_per_layer)[node.name])

    return dict(cycles_rtlsim=cycles_rtlsim, exp_cycles=exp_cycles,
                odim=[odim_h, odim_w], shape_ok=shape_ok, values_ok=values_ok)


def main():
    done = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            done = {r["name"]: r for r in json.load(f)}
        print(f"resuming: {len(done)} config(s) already in {RESULTS}")

    results = list(done.values())
    for (name, K, S, P, H, W, CI, CO, PE, SIMD) in CONFIGS:
        if name in done:
            print(f"[skip] {name} (already measured)")
            continue
        cfg = dict(name=name, K=K, S=S, P=P, H=H, W=W, CI=CI, CO=CO, PE=PE, SIMD=SIMD)
        print(f"\n=== {name}: K={K} S={S} P={P} H={H} W={W} CI={CI} CO={CO} "
              f"PE={PE} SIMD={SIMD} ===", flush=True)
        try:
            cfg.update(run_one(K, S, P, H, W, CI, CO, PE, SIMD))
            d = cfg["cycles_rtlsim"] - cfg["exp_cycles"]
            pct = 100.0 * d / cfg["cycles_rtlsim"] if cfg["cycles_rtlsim"] else float("nan")
            cfg["ok"] = True
            print(f"  rtlsim={cfg['cycles_rtlsim']}  exp={cfg['exp_cycles']}  "
                  f"rtlsim-exp={d:+d} ({pct:+.2f}%)  "
                  f"shape_ok={cfg['shape_ok']} values_ok={cfg['values_ok']}", flush=True)
        except Exception as e:  # keep going; one bad config must not lose the rest
            cfg["ok"] = False
            cfg["error"] = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {cfg['error']}", flush=True)
            traceback.print_exc()

        results.append(cfg)
        with open(RESULTS, "w") as f:  # write after every config
            json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print(f"{'name':<14}{'K':>2}{'P':>2}{'H':>4}{'W':>4}{'PE':>3}{'SIMD':>5} | "
          f"{'rtlsim':>8}{'exp':>8}{'diff':>7}{'diff%':>8}")
    print("=" * 78)
    diffs, pcts = [], []
    for r in results:
        if not r.get("ok"):
            print(f"{r['name']:<14} FAILED: {r.get('error', '')[:50]}")
            continue
        d = r["cycles_rtlsim"] - r["exp_cycles"]
        pct = 100.0 * d / r["cycles_rtlsim"]
        diffs.append(d)
        pcts.append(pct)
        flag = "" if (r["shape_ok"] and r["values_ok"]) else "  <-- FUNCTIONAL MISMATCH"
        print(f"{r['name']:<14}{r['K']:>2}{r['P']:>2}{r['H']:>4}{r['W']:>4}"
              f"{r['PE']:>3}{r['SIMD']:>5} | {r['cycles_rtlsim']:>8}{r['exp_cycles']:>8}"
              f"{d:>+7}{pct:>+7.2f}%{flag}")
    if diffs:
        n = len(diffs)
        print("-" * 78)
        print(f"n = {n}")
        print(f"mean (rtlsim - exp)      : {sum(diffs)/n:+.1f} cycles")
        print(f"mean |rtlsim - exp|      : {sum(abs(d) for d in diffs)/n:.1f} cycles")
        print(f"mean signed  error       : {sum(pcts)/n:+.3f} %")
        print(f"mean absolute error      : {sum(abs(p) for p in pcts)/n:.3f} %")
        print(f"max  absolute error      : {max(abs(d) for d in diffs)} cycles "
              f"({max(abs(p) for p in pcts):.3f} %)")
    print(f"\nresults written to {RESULTS}")


if __name__ == "__main__":
    main()
