# Jude: Created MM2IMv2
"""Tests for DeconvolutionMM2IM (finn-hlslib mm2im.hpp) and InferDeconvolution(impl="mm2im").

Fast tests (no Vivado): conversion, numpy execution, cycle model, accumulator minimisation,
resource estimates. Slow tests: cppsim and rtlsim (bit-exact vs onnxruntime ConvTranspose,
rtlsim cycles vs get_exp_cycles) over geometries rev2d cannot do (K % S != 0, K < S) as well as
the ESPCN deconv shape, with SKIP on/off and the minimised accumulator.
"""
import pytest

import numpy as np
import warnings
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.infer_deconvolution import InferDeconvolution
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import pynq_part_map

test_fpga_part = pynq_part_map["AUP-ZU3_8GB"]
target_clk_ns = 10


def make_convtranspose_model(idt, wdt, k, s, p, idim, ifm_ch, ofm_ch, seed=0):
    """Single ConvTranspose (NCHW, no bias) with random weights of datatype wdt."""
    idim_h, idim_w = idim
    odim_h, odim_w = (idim_h - 1) * s - 2 * p + k, (idim_w - 1) * s - 2 * p + k
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, idim_h, idim_w])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm_ch, odim_h, odim_w])
    node = helper.make_node(
        "ConvTranspose",
        ["inp", "W"],
        ["outp"],
        dilations=(1, 1),
        group=1,
        kernel_shape=(k, k),
        pads=(p, p, p, p),
        strides=(s, s),
    )
    graph = helper.make_graph([node], "convtranspose_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="convtranspose-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("W", wdt)
    np.random.seed(seed)
    model.set_initializer("W", gen_finn_dt_tensor(wdt, [ifm_ch, ofm_ch, k, k]))
    model = model.transform(InferShapes())
    return model


def to_mm2im(ref_model, pe, simd, skip=1, minimize=True, specialize=True,
             mem_mode="internal_embedded"):
    """InferDeconvolution(impl="mm2im"), optional MinimizeAccumulatorWidth, folding, and
    SpecializeLayers (the HLS op only executes in cppsim / rtlsim, so python execution needs
    specialize=False)."""
    model = ref_model.transform(InferDeconvolution(impl="mm2im", mem_mode=mem_mode))
    assert [n.op_type for n in model.graph.node] == ["Transpose", "DeconvolutionMM2IM", "Transpose"]
    if minimize:
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(InferDataTypes())
    if specialize:
        model = model.transform(SpecializeLayers(test_fpga_part))
    op_type = "DeconvolutionMM2IM_hls" if specialize else "DeconvolutionMM2IM"
    inst = getCustomOp(model.get_nodes_by_op_type(op_type)[0])
    inst.set_nodeattr("PE", pe)
    inst.set_nodeattr("SIMD", simd)
    if mem_mode == "internal_embedded":
        inst.set_nodeattr("SKIP", skip)
    model = model.transform(GiveUniqueNodeNames())  # copies the model: fetch inst again
    return model, getCustomOp(model.get_nodes_by_op_type(op_type)[0])


def assert_cycles(measured, expected, rows):
    """The model's accuracy: the compute pipeline depth varies by configuration (5-8), which
    leaves up to ~6 cycles per input row unmodelled; negligible at real image sizes."""
    assert abs(measured - expected) <= max(0.03 * expected, 6 * rows), (measured, expected)


# idt, K, S, P, idim, CI, CO, SIMD, PE
GEOMETRIES = {
    "espcn": ("UINT4", 6, 2, 2, [8, 8], 8, 3, 4, 3),  # ESPCN deconv shape, small image
    "k_mod_s": ("INT4", 3, 2, 1, [6, 6], 4, 3, 4, 1),  # K % S != 0 (rev2d cannot)
    "k_lt_s": ("UINT4", 2, 3, 0, [5, 4], 2, 2, 1, 1),  # K < S, non-square (rev2d cannot)
    "s1": ("INT8", 3, 1, 1, [5, 5], 2, 3, 2, 1),
    "p_k_m1": ("UINT8", 5, 2, 4, [6, 6], 2, 2, 2, 1),  # P = K-1
    "hazard": ("UINT4", 2, 1, 0, [5, 5], 1, 1, 1, 1),  # shortest RMW same-address distance
}


@pytest.mark.parametrize("geom", list(GEOMETRIES))
@pytest.mark.fpgadataflow
def test_mm2im_infer_and_python_exec(geom):
    idt_name, k, s, p, idim, ifm_ch, ofm_ch, simd, pe = GEOMETRIES[geom]
    idt, wdt = DataType[idt_name], DataType["INT8"]
    ref_model = make_convtranspose_model(idt, wdt, k, s, p, idim, ifm_ch, ofm_ch)
    x = gen_finn_dt_tensor(idt, [1, ifm_ch] + idim)
    y_expected = oxe.execute_onnx(ref_model, {"inp": x})["outp"]

    # rev2d leaves K % S != 0 alone, mm2im converts everything here
    if k % s != 0:
        with pytest.warns(UserWarning, match="Can't infer Deconvolution"):
            rev2d = ref_model.transform(InferDeconvolution())
        assert [n.op_type for n in rev2d.graph.node] == ["ConvTranspose"]

    model = ref_model.transform(InferDeconvolution(impl="mm2im"))
    assert [n.op_type for n in model.graph.node] == ["Transpose", "DeconvolutionMM2IM", "Transpose"]
    assert (oxe.execute_onnx(model, {"inp": x})["outp"] == y_expected).all()

    # the minimised accumulator still holds the exact result, and is narrower than INT32;
    # the output follows it (the Transpose after it keeps it from being the graph output)
    model, inst = to_mm2im(ref_model, pe, simd, specialize=False)
    adt = inst.get_accumulator_datatype()
    assert adt.bitwidth() < 32
    assert inst.get_output_datatype() == adt
    assert model.get_tensor_datatype("outp") == adt
    assert adt.min() <= y_expected.min() and y_expected.max() <= adt.max()
    assert (oxe.execute_onnx(model, {"inp": x})["outp"] == y_expected).all()


@pytest.mark.fpgadataflow
def test_mm2im_acc_bound_is_tight():
    # one output channel, one phase per output (S = K): the bound is reached by an input at
    # the extreme matching each weight's sign
    idt, wdt = DataType["UINT4"], DataType["INT4"]
    ref_model = make_convtranspose_model(idt, wdt, 2, 2, 0, [3, 3], 4, 1, seed=3)
    model, inst = to_mm2im(ref_model, 1, 1)
    w = model.get_initializer(inst.onnx_node.input[1])
    lo, hi = inst._acc_range(w, idt)
    exp_hi = max(np.where(w[0, ky, kx] > 0, w[0, ky, kx] * idt.max(), 0).sum()
                 for ky in range(2) for kx in range(2))
    assert hi == exp_hi
    adt = inst.get_accumulator_datatype()
    assert adt.min() <= lo and hi <= adt.max()


@pytest.mark.parametrize(
    "geom,skip,exp",
    [
        # cosim-validated points of finn-hlslib tb/run_mm2im_cosim.sh
        # (iterations + 21*H - 34 with SKIP, 17*H - 34 without)
        ("espcn", 1, 4128 + 134),
        ("espcn", 0, 4864 + 102),
        ("k_lt_s", 1, 628 + 71),
        ("hazard", 1, 136 + 71),
    ],
)
@pytest.mark.fpgadataflow
def test_mm2im_exp_cycles(geom, skip, exp):
    idt_name, k, s, p, idim, ifm_ch, ofm_ch, simd, pe = GEOMETRIES[geom]
    ref_model = make_convtranspose_model(
        DataType[idt_name], DataType["INT8"], k, s, p, idim, ifm_ch, ofm_ch
    )
    inst = to_mm2im(ref_model, pe, simd, skip=skip, minimize=False)[1]
    assert inst.get_exp_cycles() == exp


@pytest.mark.parametrize(
    "simd,pe,acc_bits,exp_bram",
    [
        # ESPCN deconv K6 S2 P2 128x128, CI32 CO3: accumulator 3 bands x 2 rows x 256 x CF
        (4, 3, 17, 3 * 1 * 2),  # 1536 x 17 per lane -> 2 BRAM18, weight ROM 288 x 8 -> BRAM
        (4, 3, 32, 3 * 2 * 2),
    ],
)
@pytest.mark.fpgadataflow
def test_mm2im_resource_estimates(simd, pe, acc_bits, exp_bram):
    ref_model = make_convtranspose_model(
        DataType["UINT4"], DataType["INT8"], 6, 2, 2, [128, 128], 32, 3
    )
    inst = to_mm2im(ref_model, pe, simd, minimize=False)[1]
    inst.set_nodeattr("accDataType", "INT%d" % acc_bits)
    inst.set_nodeattr("ram_style", "distributed")
    assert inst.bram_estimation(test_fpga_part) == exp_bram
    assert inst.dsp_estimation(test_fpga_part) == pe * simd
    inst.set_nodeattr("ram_style", "block")
    assert inst.bram_estimation(test_fpga_part) == exp_bram + pe * simd


@pytest.mark.parametrize(
    "target_cycles,wwidth_max,exp_simd,exp_pe",
    [
        # ESPCN deconv K6 S2 P2 128x128, CI32 CO3; MM2IM with SKIP needs ~56.2 M cycles at
        # PE1/SIMD1, so the same folding as rev2d (test_set_folding_deconv) meets each target
        (10**9, 36, 1, 1),  # PE1/SIMD1 already meets the target
        (4 * 10**7, 36, 2, 1),  # SIMD 2 is enough (~28.2 M cycles)
        (1, 36, 4, 3),  # maxed: 8-bit weights x SIMD 4 = 32 <= 36, all 3 output channels
        (1, 256, 32, 3),  # wider weight words allowed: SIMD up to all 32 input channels
    ],
)
@pytest.mark.fpgadataflow
def test_set_folding_mm2im(target_cycles, wwidth_max, exp_simd, exp_pe):
    ref_model = make_convtranspose_model(
        DataType["UINT4"], DataType["INT8"], 6, 2, 2, [128, 128], 32, 3
    )
    # SetFolding needs a graph of HW layers only: drop InferDeconvolution's Transposes
    model = make_mm2im_hw_model(ref_model, 1, 1, "internal_embedded")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*bottleneck.*")
        model = model.transform(
            SetFolding(target_cycles_per_frame=target_cycles, mvau_wwidth_max=wwidth_max)
        )
    inst = getCustomOp(model.get_nodes_by_op_type("DeconvolutionMM2IM_hls")[0])
    assert (inst.get_nodeattr("SIMD"), inst.get_nodeattr("PE")) == (exp_simd, exp_pe)
    if exp_simd < 32 and target_cycles > 1:
        assert inst.get_nodeattr("cycles_estimate") < target_cycles


@pytest.mark.parametrize(
    "geom,skip,minimize",
    [(g, 1, True) for g in GEOMETRIES] + [("espcn", 0, True), ("espcn", 1, False)],
)
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_deconv_mm2im(geom, skip, minimize, exec_mode):
    idt_name, k, s, p, idim, ifm_ch, ofm_ch, simd, pe = GEOMETRIES[geom]
    idt, wdt = DataType[idt_name], DataType["INT8"]
    ref_model = make_convtranspose_model(idt, wdt, k, s, p, idim, ifm_ch, ofm_ch)
    x = gen_finn_dt_tensor(idt, [1, ifm_ch] + idim)
    y_expected = oxe.execute_onnx(ref_model, {"inp": x})["outp"]

    model = to_mm2im(ref_model, pe, simd, skip=skip, minimize=minimize)[0]
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    else:
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
        model = model.transform(SetExecMode("rtlsim"))

    y_produced = oxe.execute_onnx(model, {"inp": x})["outp"]
    assert y_produced.shape == y_expected.shape
    assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("DeconvolutionMM2IM_hls")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        print("mm2im %s: expected cycles %d, rtlsim %d" % (geom, exp_cycles, cycles_rtlsim))
        assert_cycles(cycles_rtlsim, exp_cycles, idim[0])


def sweep_geometries():
    """The plan's P3 sweep: K 2..5 x S 1..3 x P in {0, 1, K-S} (where 0 <= P < K), on a
    5x4 input. Input datatype, PE and SIMD rotate through sub-byte and byte types and
    1/2-way parallelism so every combination is covered somewhere."""
    idts = ["UINT4", "INT4", "INT2", "UINT8", "INT8", "UINT2"]
    out, i = [], 0
    for k in range(2, 6):
        for s in range(1, 4):
            for p in sorted({0, 1, k - s}):
                if p < 0 or p >= k or 2 * p >= 3 * s + k:  # (4-1)*S + K: the smaller side
                    continue
                out.append(
                    pytest.param(
                        idts[i % 6], k, s, p, [5, 4], 4, 2, (1, 2, 4)[i % 3], (1, 2)[i % 2],
                        id="k%ds%dp%d" % (k, s, p),
                    )
                )
                i += 1
    return out


@pytest.mark.parametrize("idt_name,k,s,p,idim,ifm_ch,ofm_ch,simd,pe", sweep_geometries())
@pytest.mark.parametrize("skip", [1, 0])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_deconv_mm2im_sweep(
    idt_name, k, s, p, idim, ifm_ch, ofm_ch, simd, pe, skip, exec_mode
):
    idt, wdt = DataType[idt_name], DataType["INT4"]
    ref_model = make_convtranspose_model(idt, wdt, k, s, p, idim, ifm_ch, ofm_ch, seed=k * 100 + s * 10 + p)
    x = gen_finn_dt_tensor(idt, [1, ifm_ch] + idim)
    y_expected = oxe.execute_onnx(ref_model, {"inp": x})["outp"]

    model, inst = to_mm2im(ref_model, pe, simd, skip=skip)
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    else:
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
        model = model.transform(SetExecMode("rtlsim"))
    y_produced = oxe.execute_onnx(model, {"inp": x})["outp"]
    assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("DeconvolutionMM2IM_hls")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        print("mm2im k%ds%dp%d skip%d: expected cycles %d, rtlsim %d"
              % (k, s, p, skip, exp_cycles, cycles_rtlsim))
        assert_cycles(cycles_rtlsim, exp_cycles, idim[0])


@pytest.mark.parametrize(
    "k,stride,padding,group,impl,converts",
    [
        (4, 2, 1, 1, "rev2d", True),
        (4, 2, 1, 1, "mm2im", True),
        (5, 2, 1, 1, "rev2d", False),  # stride does not divide the kernel: rev2d only
        (5, 2, 1, 1, "mm2im", True),
        (2, 3, 0, 1, "rev2d", False),  # K < S: rev2d only
        (2, 3, 0, 1, "mm2im", True),
        (4, 2, 1, 2, "rev2d", False),  # grouped: neither
        (4, 2, 1, 2, "mm2im", False),
        (2, 1, 2, 1, "mm2im", False),  # P >= K: mm2im refuses it
    ],
)
@pytest.mark.fpgadataflow
def test_infer_deconv_impl_support(k, stride, padding, group, impl, converts):
    ifm_ch, ofm_ch, idim = 4, 4, 6
    odim = (idim - 1) * stride - 2 * padding + k
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, idim, idim])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm_ch, odim, odim])
    node = helper.make_node(
        "ConvTranspose",
        ["inp", "W"],
        ["outp"],
        group=group,
        kernel_shape=(k, k),
        pads=(padding,) * 4,
        strides=(stride, stride),
    )
    graph = helper.make_graph([node], "convtranspose_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="convtranspose-model"))
    model.set_tensor_datatype("inp", DataType["UINT4"])
    model.set_tensor_datatype("W", DataType["INT8"])
    model.set_initializer(
        "W", gen_finn_dt_tensor(DataType["INT8"], [ifm_ch, ofm_ch // group, k, k])
    )
    model = model.transform(InferShapes())
    op_type = {"rev2d": "Deconvolution", "mm2im": "DeconvolutionMM2IM"}[impl]
    if converts:
        model = model.transform(InferDeconvolution(impl=impl))
        assert [n.op_type for n in model.graph.node] == ["Transpose", op_type, "Transpose"]
    else:
        with pytest.warns(UserWarning, match="Can't infer Deconvolution"):
            model = model.transform(InferDeconvolution(impl=impl))
        assert [n.op_type for n in model.graph.node] == ["ConvTranspose"]


# -- P2: streamed weights (mem_mode internal_decoupled / external) ---------------------


@pytest.mark.parametrize("mem_mode", ["internal_decoupled", "external"])
@pytest.mark.fpgadataflow
def test_mm2im_streamed_attrs(mem_mode, tmp_path):
    idt_name, k, s, p, idim, ifm_ch, ofm_ch, simd, pe = GEOMETRIES["espcn"]
    ref_model = make_convtranspose_model(
        DataType[idt_name], DataType["INT8"], k, s, p, idim, ifm_ch, ofm_ch
    )
    model, inst = to_mm2im(ref_model, pe, simd, mem_mode=mem_mode)
    assert inst.get_nodeattr("SKIP") == 0 and inst.get_skip() == 0
    wmem = k * k * (ofm_ch // pe) * (ifm_ch // simd)
    assert inst.calc_wmem() == wmem
    assert inst.get_instream_width(1) == pe * simd * 8
    assert inst.get_folded_input_shape(1) == (1, idim[0] * idim[1] * wmem, pe * simd)
    assert inst.get_exp_cycles() == 4864 + 102  # cosim-validated, as SKIP=0
    inst.set_nodeattr("SKIP", 1)  # ignored, and reported
    assert inst.get_skip() == 0 and any("SKIP is ignored" in m for m in inst.verify_node())
    s_axis = [i[0] for i in inst.get_verilog_top_module_intf_names()["s_axis"]]
    assert s_axis == (["in0_V"] if mem_mode == "internal_decoupled" else ["in0_V", "in1_V"])

    # weight stream: word t = ((ky*K + kx)*CF + cf)*SF + sf, element pe*SIMD + simd
    w = model.get_initializer(inst.onnx_node.input[1])
    words = inst.get_weight_stream_words(w)
    cf_n, sf_n = ofm_ch // pe, ifm_ch // simd
    for ky, kx, cf, sf, q, r in [(0, 0, 0, 0, 0, 0), (5, 3, 0, 1, 2, 3), (2, 4, 0, 0, 1, 2)]:
        t = ((ky * k + kx) * cf_n + cf) * sf_n + sf
        assert words[t, q * simd + r] == w[cf * pe + q, ky, kx, sf * simd + r]

    # memstream contents: element 0 in the LSBs (hls::vector compact=bit)
    dat = tmp_path / "memblock.dat"
    inst.make_weight_file(w, "decoupled_verilog_dat", str(dat))
    lines = dat.read_text().split()
    assert len(lines) == wmem
    word = int(lines[7], 16)
    for e in range(pe * simd):
        v = (word >> (8 * e)) & 0xFF
        assert (v - 256 if v > 127 else v) == words[7, e]


@pytest.mark.parametrize("geom", ["espcn", "k_mod_s", "k_lt_s", "hazard"])
@pytest.mark.parametrize("mem_mode", ["internal_decoupled", "external"])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_deconv_mm2im_streamed(geom, mem_mode, exec_mode):
    idt_name, k, s, p, idim, ifm_ch, ofm_ch, simd, pe = GEOMETRIES[geom]
    idt, wdt = DataType[idt_name], DataType["INT8"]
    ref_model = make_convtranspose_model(idt, wdt, k, s, p, idim, ifm_ch, ofm_ch)
    x = gen_finn_dt_tensor(idt, [1, ifm_ch] + idim)
    y_expected = oxe.execute_onnx(ref_model, {"inp": x})["outp"]

    model = to_mm2im(ref_model, pe, simd, mem_mode=mem_mode)[0]
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    else:
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
        model = model.transform(SetExecMode("rtlsim"))
    y_produced = oxe.execute_onnx(model, {"inp": x})["outp"]
    assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("DeconvolutionMM2IM_hls")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        print("mm2im %s %s: expected cycles %d, rtlsim %d"
              % (geom, mem_mode, exp_cycles, cycles_rtlsim))
        assert_cycles(cycles_rtlsim, exp_cycles, idim[0])


def make_mm2im_hw_model(ref_model, pe, simd, mem_mode):
    """The DeconvolutionMM2IM_hls node alone (NHWC in and out), for a stitched IP: the
    Transposes InferDeconvolution puts around it have no HW implementation."""
    model, inst = to_mm2im(ref_model, pe, simd, mem_mode=mem_mode)
    node = inst.onnx_node
    model.graph.input[0].type.tensor_type.shape.CopyFrom(
        model.get_tensor_valueinfo(node.input[0]).type.tensor_type.shape
    )
    model.graph.output[0].type.tensor_type.shape.CopyFrom(
        model.get_tensor_valueinfo(node.output[0]).type.tensor_type.shape
    )
    gin, gout = model.graph.input[0].name, model.graph.output[0].name
    node.input[0], node.output[0] = gin, gout
    for t in [n for n in model.graph.node if n.op_type == "Transpose"]:
        model.graph.node.remove(t)
    model.set_tensor_datatype(gin, inst.get_input_datatype(0))
    model.set_tensor_datatype(gout, inst.get_output_datatype())
    model = model.transform(InferShapes())
    return model.transform(GiveUniqueNodeNames())


@pytest.mark.parametrize(
    "geom,mem_mode",
    [
        ("espcn", "internal_decoupled"),
        ("k_mod_s", "internal_decoupled"),
        ("hazard", "internal_decoupled"),
        ("espcn", "internal_embedded"),  # control: same flow without a memstream
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_deconv_mm2im_stitched(geom, mem_mode):
    """Stitched-IP rtlsim, so the memstream (internal_decoupled) is in the loop, after FIFO
    sizing by rtlsim characterisation (derive_characteristic_fxns with the weight input)."""
    idt_name, k, s, p, idim, ifm_ch, ofm_ch, simd, pe = GEOMETRIES[geom]
    idt, wdt = DataType[idt_name], DataType["INT8"]
    ref_model = make_convtranspose_model(idt, wdt, k, s, p, idim, ifm_ch, ofm_ch)
    x = gen_finn_dt_tensor(idt, [1, ifm_ch] + idim)
    y_expected = oxe.execute_onnx(ref_model, {"inp": x})["outp"]

    model = make_mm2im_hw_model(ref_model, pe, simd, mem_mode)
    model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")

    gin, gout = model.graph.input[0].name, model.graph.output[0].name
    y_produced = oxe.execute_onnx(model, {gin: x.transpose(0, 2, 3, 1)})[gout]
    assert (y_produced.transpose(0, 3, 1, 2) == y_expected).all()
