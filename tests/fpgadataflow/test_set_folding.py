# Copyright (c) 2020, Xilinx
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
# * Neither the name of FINN nor the names of its
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
# Jude: Edited
import warnings
# Jude: Done
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.util.test import load_test_checkpoint_or_skip


def make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes):
    W = np.random.randint(wdt.min(), wdt.max() + 1, size=(ch, ch))
    W = W.astype(np.float32)

    T = np.random.randint(tdt.min(), tdt.max() + 1, size=(ch, 2 ** adt.bitwidth() - 1))
    T = T.astype(np.float32)

    tensors = []
    tensors.append(helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch]))
    for i in range(1, nnodes):
        inter = helper.make_tensor_value_info("inter_" + str(i), TensorProto.FLOAT, [1, ch])
        tensors.append(inter)
    tensors.append(helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch]))

    FCLayer_nodes = []
    for i in range(nnodes):
        pe = 1
        simd = 1
        FCLayer_nodes += [
            helper.make_node(
                "MVAU_hls",
                [tensors[i].name, "weights_" + str(i), "thresh_" + str(i)],
                [tensors[i + 1].name],
                domain="finn.custom_op.fpgadataflow.hls",
                backend="fpgadataflow",
                MW=ch,
                MH=ch,
                SIMD=simd,
                PE=pe,
                inputDataType=adt.name,
                weightDataType=wdt.name,
                outputDataType=adt.name,
                ActVal=0,
                binaryXnorMode=0,
                noActivation=0,
            )
        ]

    graph = helper.make_graph(
        nodes=FCLayer_nodes,
        name="fclayer_graph",
        inputs=[tensors[0]],
        outputs=[tensors[-1]],
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", adt)
    model.set_tensor_datatype("outp", adt)

    for i in range(1, nnodes + 1):
        if tensors[i].name != "outp":
            model.graph.value_info.append(tensors[i])
        model.set_initializer("weights_" + str(i - 1), W)
        model.set_initializer("thresh_" + str(i - 1), T)
        model.set_tensor_datatype("weights_" + str(i - 1), wdt)
        model.set_tensor_datatype("thresh_" + str(i - 1), tdt)

    return model


# desired frames per second
@pytest.mark.parametrize("target_fps", [30, 10**5, 10**7])
# target chip or board
@pytest.mark.parametrize("platform", ["AUP-ZU3_8GB", "Ultra96", "U200"])
@pytest.mark.fpgadataflow
def test_set_folding(target_fps, platform):
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 5)

    model = model.transform(GiveUniqueNodeNames())
    parent_model = model.transform(CreateDataflowPartition())
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    clk_ns = 5
    target_cycles_per_frame = int((10**9 / clk_ns) / target_fps)
    dataflow_model = dataflow_model.transform(SetFolding(target_cycles_per_frame))

    exp_cycles_dict = dataflow_model.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame = max(exp_cycles_dict.values())

    min_cycles = dict()
    min_cycles["AUP-ZU3_8GB"] = 64
    min_cycles["Ultra96"] = 64
    min_cycles["U200"] = 1

    assert achieved_cycles_per_frame <= max(
        min_cycles[platform], target_cycles_per_frame
    ), "Folding target not met"
# Jude: Edited


def make_deconv_model(wdt=DataType["INT8"]):
    # ESPCN's ConvTranspose as a Deconvolution_hls node: K=6, S=2, P=2, 128x128x32 -> 256x256x3
    k, s, p, idim, ifm_ch, ofm_ch = 6, 2, 2, 128, 32, 3
    odim = (idim - 1) * s - 2 * p + k
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, idim, idim, ifm_ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim, odim, ofm_ch])
    node = helper.make_node(
        "Deconvolution_hls",
        ["inp", "W"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        KernelDim=[k, k],
        IFMChannels=ifm_ch,
        OFMChannels=ofm_ch,
        IFMDim=[idim, idim],
        Stride=[s, s],
        Padding=[p, p],
        PE=1,
        SIMD=1,
        inputDataType="UINT4",
        weightDataType=wdt.name,
        outputDataType="INT32",
        cpp_interface="hls_vector",
        hls_style="freerunning",
    )
    graph = helper.make_graph([node], "deconv_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="deconv-model"))
    model.set_initializer("W", np.zeros((ofm_ch, k, k, ifm_ch), dtype=np.float32))
    model.set_tensor_datatype("W", wdt)
    return model.transform(GiveUniqueNodeNames())


@pytest.mark.parametrize(
    "target_cycles,wwidth_max,exp_simd,exp_pe",
    [
        (10**9, 36, 1, 1),  # PE1/SIMD1 (~56.6 M cycles) already meets the target
        (4 * 10**7, 36, 2, 1),  # SIMD 2 is enough
        (1, 36, 4, 3),  # maxed: 8-bit weights x SIMD 4 = 32 <= 36, all 3 output channels
        (1, 256, 32, 3),  # wider weight words allowed: SIMD up to all 32 input channels
    ],
)
@pytest.mark.fpgadataflow
def test_set_folding_deconv(target_cycles, wwidth_max, exp_simd, exp_pe):
    model = make_deconv_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*get_exp_cycles.*")
        warnings.filterwarnings("ignore", message=".*bottleneck.*")
        model = model.transform(
            SetFolding(target_cycles_per_frame=target_cycles, mvau_wwidth_max=wwidth_max)
        )
    inst = getCustomOp(model.graph.node[0])
    assert (inst.get_nodeattr("SIMD"), inst.get_nodeattr("PE")) == (exp_simd, exp_pe)
    if exp_simd < 32 and target_cycles > 1:
        assert inst.get_nodeattr("cycles_estimate") < target_cycles
# Jude: Done
