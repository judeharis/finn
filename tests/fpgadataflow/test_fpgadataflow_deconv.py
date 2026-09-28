# Copyright (c) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferConvInpGen,
    InferQuantizedMatrixVectorActivation,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import (
    InferPixelPaddingDeconv,
)
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import pynq_part_map

test_pynq_board = "Pynq-Z1"
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def set_up_reference_model(idt, wdt, odt, k, idim, ifm_ch, ofm_ch, stride, padding):
    idim_h, idim_w = idim
    stride_h, stride_w = stride
    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    inp = helper.make_tensor_value_info(
        "inp",
        TensorProto.FLOAT,
        [
            1,
            ifm_ch,
            idim_h,
            idim_w,
        ],
    )
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm_ch, odim_h, odim_w])

    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [ifm_ch, ofm_ch, k, k])

    ConvTranspose = helper.make_node(
        "ConvTranspose",
        ["inp", "W"],
        ["outp"],
        dilations=(1, 1),
        group=1,
        kernel_shape=(k, k),
        pads=(padding, padding, padding, padding),
        strides=(stride_h, stride_w),
    )

    node_list = [ConvTranspose]
    value_info = [W]

    graph = helper.make_graph(
        nodes=node_list,
        name="convtranspose_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=value_info,
    )

    model = qonnx_make_model(graph, producer_name="convtranspose-model")
    model = ModelWrapper(model)

    # initialize model
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype(model.get_first_global_out(), odt)
    model.set_tensor_datatype("W", wdt)

    w_tensor = gen_finn_dt_tensor(wdt, [ifm_ch, ofm_ch, k, k])
    model.set_initializer("W", w_tensor)

    model = model.transform(InferShapes())

    return model, w_tensor


def create_deconv_node(idt, wdt, odt, k, idim, ifm_ch, ofm_ch, stride, padding, w_tensor):
    idim_h, idim_w = idim
    stride_h, stride_w = stride
    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    inp = helper.make_tensor_value_info(
        "inp",
        TensorProto.FLOAT,
        [
            1,
            idim_h,
            idim_w,
            ifm_ch,
        ],
    )
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim_h, odim_w, ofm_ch])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [ofm_ch, k, k, ifm_ch])

    Deconv = helper.make_node(
        "Deconvolution_hls",
        ["inp", "W"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        KernelDim=[k, k],
        IFMChannels=ifm_ch,
        OFMChannels=ofm_ch,
        IFMDim=idim,
        Stride=[stride_h, stride_w],
        Padding=[padding, padding],
        PE=1,
        SIMD=1,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        cpp_interface="hls_vector",
        hls_style="freerunning",
    )

    node_list = [Deconv]
    value_info = [W]

    graph = helper.make_graph(
        nodes=node_list,
        name="convtranspose_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=value_info,
    )

    model = qonnx_make_model(graph, producer_name="convtranspose-model")
    model = ModelWrapper(model)

    # initialize model
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype(model.graph.output[0].name, odt)
    model.set_tensor_datatype("W", wdt)

    w_tensor = w_tensor.transpose(1, 2, 3, 0)
    model.set_initializer("W", w_tensor)

    model = model.transform(InferShapes())

    return model


# input image dimension
@pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of rows and number of cols to add
@pytest.mark.parametrize("stride", [[2, 2], [2, 3]])
# number of channels
@pytest.mark.parametrize("ifm_ch", [2])
# number of channels
@pytest.mark.parametrize("ofm_ch", [4])
# Input parallelism
@pytest.mark.parametrize("simd", [1, 2])
# PE
@pytest.mark.parametrize("pe", [1, 2])
# kernel size
@pytest.mark.parametrize("k", [2])
# padding
@pytest.mark.parametrize("padding", [0, 1])
# exec mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_deconv_pixel_pad(
    idim, stride, ifm_ch, ofm_ch, simd, pe, k, padding, exec_mode
):
    idt = wdt = DataType["INT4"]
    wdt = idt
    odt = DataType["INT32"]
    idim_h, idim_w = idim
    stride_h, stride_w = stride

    ref_model = set_up_reference_model(idt, wdt, odt, k, idim, ifm_ch, ofm_ch, stride, padding)[0]

    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, idim_h, idim_w])
    input_dict = {"inp": input_tensor}

    y_expected = oxe.execute_onnx(ref_model, input_dict)["outp"]

    model = ref_model.transform(InferPixelPaddingDeconv())
    model = model.transform(InferConvInpGen())
    model = model.transform(InferQuantizedMatrixVectorActivation())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(MinimizeAccumulatorWidth())

    for n in model.graph.node:
        if n.op_type.startswith("ConvolutionInputGenerator"):
            convinputgen_node = getCustomOp(n)
            convinputgen_node.set_nodeattr("SIMD", simd)
        elif n.op_type.startswith("MVAU"):
            mvau_node = getCustomOp(n)
            mvau_node.set_nodeattr("PE", pe)
            mvau_node.set_nodeattr("SIMD", simd)

    expected_oshape = (1, ofm_ch, odim_h, odim_w)

    # cppsim
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))

    # rtlsim
    else:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
        model = model.transform(SetExecMode("rtlsim"))

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert y_produced.shape == expected_oshape
    assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("FMPadding_Pixel_hls")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


# input image dimension
@pytest.mark.parametrize("idim", [[8, 8]])
# number of rows and number of cols to add
@pytest.mark.parametrize("stride", [[2, 2]])
# number of channels
@pytest.mark.parametrize("ifm_ch", [2])
# number of channels
@pytest.mark.parametrize("ofm_ch", [3])
# Input parallelism
@pytest.mark.parametrize("simd", [1])
# PE
@pytest.mark.parametrize("pe", [1])
# kernel size
@pytest.mark.parametrize("k", [4])
# padding
@pytest.mark.parametrize("padding", [1])
# exec mode
# Jude: Edited
@pytest.mark.parametrize("exec_mode", ["rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado

# # input image dimension
# @pytest.mark.parametrize("idim", [[128, 128]])
# # number of rows and number of cols to add
# @pytest.mark.parametrize("stride", [[2, 2]])
# # number of channels
# @pytest.mark.parametrize("ifm_ch", [32])
# # number of channels
# @pytest.mark.parametrize("ofm_ch", [3])
# # Input parallelism
# @pytest.mark.parametrize("simd", [1])
# # PE
# @pytest.mark.parametrize("pe", [1])
# # kernel size
# @pytest.mark.parametrize("k", [6])
# # padding
# @pytest.mark.parametrize("padding", [2])
# # exec mode
# @pytest.mark.parametrize("exec_mode", ["cppsim"])
# @pytest.mark.fpgadataflow
# @pytest.mark.slow
# @pytest.mark.vivado
# Jude: Done
def test_fpgadataflow_deconv_revd2(idim, stride, ifm_ch, ofm_ch, simd, pe, k, padding, exec_mode):
    idt = wdt = DataType["INT8"]
    wdt = idt
    odt = DataType["INT32"]
    idim_h, idim_w = idim
    stride_h, stride_w = stride

    ref_model, w_tensor = set_up_reference_model(
        idt, wdt, odt, k, idim, ifm_ch, ofm_ch, stride, padding
    )
    model = create_deconv_node(idt, wdt, odt, k, idim, ifm_ch, ofm_ch, stride, padding, w_tensor)

    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, idim_h, idim_w])
    input_dict = {"inp": input_tensor}

    y_expected = oxe.execute_onnx(ref_model, input_dict)["outp"]

    # model = model.transform(InferShapes())
    # model = model.transform(GiveUniqueNodeNames())
    input_tensor_nhwc = input_tensor.transpose(0, 2, 3, 1)
    input_dict_nhwc = {"inp": input_tensor_nhwc}
    # y_produced = oxe.execute_onnx(model, input_dict_nhwc)["outp"]
    # assert (y_produced == y_expected).all()

    # model = model.transform(SpecializeLayers(test_fpga_part))
    # model = model.transform(MinimizeAccumulatorWidth())

    for n in model.graph.node:
        if n.op_type.startswith("Deconvolution_hls"):
            deconv_node = getCustomOp(n)
            deconv_node.set_nodeattr("PE", pe)
            deconv_node.set_nodeattr("SIMD", simd)

    expected_oshape = (1, odim_h, odim_w, ofm_ch)
    # model.save("deconv.onnx")
    # cppsim
    if exec_mode == "cppsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        # breakpoint()

    # rtlsim
    else:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
        model = model.transform(SetExecMode("rtlsim"))

    y_produced = oxe.execute_onnx(model, input_dict_nhwc)["outp"]
    assert y_produced.shape == expected_oshape
    y_produced = y_produced.transpose(0, 3, 1, 2)
    assert (y_produced == y_expected).all()
    # Jude: Edited
    print("Test passed for Deconvolution_hls with idim {}, stride {}, ifm_ch {}, ofm_ch {}, simd {}, pe {}, k {}, padding {}, exec_mode {}".format(
        idim, stride, ifm_ch, ofm_ch, simd, pe, k, padding, exec_mode
    ))
    if exec_mode == "cppsim":
        node = model.get_nodes_by_op_type("Deconvolution_hls")[0]
        inst = getCustomOp(node)
        cycles_estimate = inst.get_nodeattr("cycles_estimate")
        print("cycles_estimate: {}".format(cycles_estimate))
    # Jude: Done

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("Deconvolution_hls")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        # Jude: Edited
        print("Expected cycles: {}, RTL sim cycles: {}".format(exp_cycles, cycles_rtlsim))
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


# Regression test for the in0_V port width. Without
# `#pragma HLS aggregate variable=in0_V compact=bit`, HLS gives each sub-byte hls::vector
# element its own byte, so UINT4 x SIMD=4 synthesizes a 32-bit port while
# get_instream_width() (and so the stream and the driver) packs 16 bits; the kernel then
# reads [x0, x2, 0, 0]. It only shows with a sub-byte input, SIMD >= 2 and rtlsim: at INT8
# or SIMD=1 the two widths agree, and cppsim never synthesizes the port. Output-only on
# purpose -- the cycle check lives in test_fpgadataflow_deconv_revd2.
# input datatype, input parallelism, PE
@pytest.mark.parametrize(
    "idt_name,simd,pe",
    [
        ("UINT4", 2, 1),
        ("UINT4", 4, 3),
        ("INT4", 4, 3),
        # control: byte-aligned, so the widths agree even without the pragma
        ("INT8", 4, 3),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_deconv_subbyte_input(idt_name, simd, pe):
    idt = DataType[idt_name]
    wdt = DataType["INT8"]
    odt = DataType["INT32"]
    idim = [8, 8]
    stride = [2, 2]
    ifm_ch = 4
    ofm_ch = 3
    k = 4
    padding = 1
    idim_h, idim_w = idim
    stride_h, stride_w = stride

    ref_model, w_tensor = set_up_reference_model(
        idt, wdt, odt, k, idim, ifm_ch, ofm_ch, stride, padding
    )
    model = create_deconv_node(idt, wdt, odt, k, idim, ifm_ch, ofm_ch, stride, padding, w_tensor)

    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, idim_h, idim_w])
    y_expected = oxe.execute_onnx(ref_model, {"inp": input_tensor})["outp"]

    deconv_node = getCustomOp(model.get_nodes_by_op_type("Deconvolution_hls")[0])
    deconv_node.set_nodeattr("PE", pe)
    deconv_node.set_nodeattr("SIMD", simd)

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    model = model.transform(SetExecMode("rtlsim"))

    input_tensor_nhwc = input_tensor.transpose(0, 2, 3, 1)
    y_produced = oxe.execute_onnx(model, {"inp": input_tensor_nhwc})["outp"]
    assert y_produced.shape == (1, odim_h, odim_w, ofm_ch)
    y_produced = y_produced.transpose(0, 3, 1, 2)
    assert (y_produced == y_expected).all()
# Jude: Done
