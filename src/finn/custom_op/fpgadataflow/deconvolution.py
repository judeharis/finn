# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Deconvolution(HWCustomOp):
    """Abstraction layer for HW implementation of Deconvolution"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "KernelDim": ("ints", True, []),  # [H, W] = [Y, X]
            "IFMChannels": ("i", True, 0),
            "OFMChannels": ("i", True, 0),
            "IFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "PE": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "Stride": ("ints", True, [1, 1]),  # [H, W] = [Y, X]
            "Padding": ("ints", True, []),  # [H, W] = [Y, X]
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        if ind == 0:
            ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
            ifm_ch = self.get_nodeattr("IFMChannels")
            ishape = (1, ifm_dim_h, ifm_dim_w, ifm_ch)
        else:
            ifm_ch = self.get_nodeattr("IFMChannels")
            ofm_ch = self.get_nodeattr("OFMChannels")
            k_h, k_w = self.get_nodeattr("KernelDim")
            ishape = (ofm_ch, k_h, k_w, ifm_ch)
        return ishape

    def get_folded_input_shape(self, ind=0):
        if ind == 0:
            ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
            ifm_ch = self.get_nodeattr("IFMChannels")
            simd = self.get_nodeattr("SIMD")
            assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
            fold = int(ifm_ch / simd)
            folded_ishape = (1, ifm_dim_h, ifm_dim_w, fold, simd)
        else:
            folded_ishape = self.get_normal_input_shape(ind)
        return folded_ishape

    def get_normal_output_shape(self, ind=0):
        idim_h, idim_w = self.get_nodeattr("IFMDim")
        stride_h, stride_w = self.get_nodeattr("Stride")
        k_h, k_w = self.get_nodeattr("KernelDim")
        ofm_ch = self.get_nodeattr("OFMChannels")
        pad_h, pad_w = self.get_nodeattr("Padding")
        odim_h = (idim_h - 1) * stride_h - 2 * pad_h + (k_h - 1) + 1
        odim_w = (idim_w - 1) * stride_w - 2 * pad_w + (k_w - 1) + 1
        oshape = (1, odim_h, odim_w, ofm_ch)
        return oshape

    def get_folded_output_shape(self, ind=0):
        normal_oshape = self.get_normal_output_shape()
        odim_h = normal_oshape[1]
        odim_w = normal_oshape[2]
        ofm_ch = normal_oshape[3]
        pe = self.get_nodeattr("PE")
        fold = int(ofm_ch / pe)
        folded_oshape = (1, odim_h, odim_w, fold, pe)
        return folded_oshape

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape for Deconv."
        # implement tensor with correct shape
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_weight_datatype(self):
        """Returns FINN DataType of weights."""
        return DataType[self.get_nodeattr("weightDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self, ind=0):
        """Returns stream width, input and output stream width are equal for
        the sliding window function"""
        if ind == 0:
            ibits = self.get_input_datatype().bitwidth()
            simd = self.get_nodeattr("SIMD")
            ifm_ch = self.get_nodeattr("IFMChannels")
            assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
            in_width = simd * ibits
        else:
            in_width = 0
        return in_width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("PE")
        return out_width

    # Jude: Edited
    def get_exp_cycles(self) -> int:
        # Regression coefficients for overhead = c0 + c1*H_EFF + c2*SF + c3*K
        # + c4*K*H_EFF + c5*K*SF, fit against 32 real cosim runs. See
        # deconv_cycle_estimator.py's module docstring/--validate for provenance.
        # NOTE: every one of those 32 runs has S=1, and CROP is identically 0
        # whenever S==1 (see the CROP derivation below), so the fit says nothing
        # about either strided deconv or the cropped-tail effect.
        _OVERHEAD_COEFFS = dict(
            const=-1130.25,
            H_EFF=-182.25,
            SF=-16.25,
            K=461.40,
            K_H_EFF=98.05,
            K_SF=8.00,
        )
        _CALIBRATED_K = (3, 5)
        _CALIBRATED_S = (1,)
        _CALIBRATED_P = (1, 2)
        _CALIBRATED_H = (4, 8)
        _CALIBRATED_CI = (3,)
        _CALIBRATED_CO = (3,)
        kh, kw = self.get_nodeattr("KernelDim")
        sh, sw = self.get_nodeattr("Stride")
        ph, pw = self.get_nodeattr("Padding")
        if kh != kw:
            raise ValueError(f"deconv HLS kernel requires a square KernelDim, got {[kh, kw]}")
        if sh != sw:
            raise ValueError(f"deconv HLS kernel requires a square Stride, got {[sh, sw]}")
        if ph != pw:
            raise ValueError(f"deconv HLS kernel requires a square Padding, got {[ph, pw]}")

        K, S, P = kh, sh, ph
        H, W = self.get_nodeattr("IFMDim")
        CO = self.get_nodeattr("OFMChannels")
        CI = self.get_nodeattr("IFMChannels")
        PE = self.get_nodeattr("PE")
        SIMD = self.get_nodeattr("SIMD")

        if K % S != 0:
            raise ValueError("Stride must divide kernel size (K % S == 0).")
        if CO % PE != 0:
            raise ValueError("PE parallelism must divide output channel count (CO % PE == 0).")
        if CI % SIMD != 0:
            raise ValueError("SIMD parallelism must divide input channel count (CI % SIMD == 0).")

        KK = K // S
        CF = CO // PE
        SF = CI // SIMD

        # Transliteration of the constexpr block in deconv.hpp:453-459.
        PADUP = 0 if P >= K - S else (K - P - 1) // S
        CROP = S * PADUP - ((K - S) - P)
        H_EFF = PADUP + H + PADUP
        W_EFF = PADUP + W + PADUP
        HO_EFF = (H_EFF + 1) * S - K
        WO_EFF = (W_EFF + 1) * S - K

        N = KK * KK * SF

        # deconv_mvu emits HO_EFF*WO_EFF*CF elements in raster (h, w, cf) order,
        # but crop<CROP, HO_EFF, WO_EFF, CO> (deconv.hpp:33-61) only forwards
        # those with CROP <= h < HO_EFF-CROP and CROP <= w < WO_EFF-CROP. The
        # tail it produces after the last surviving element is never observed:
        # FINN's rtlsim stops counting once the final output word is received.
        # So the compute-bound term is set by the *last surviving* element, not
        # by the full pre-crop count. When CROP == 0 (always true for S == 1)
        # this reduces exactly to HO_EFF*WO_EFF*CF*N, leaving the S=1
        # calibration untouched.
        h_last = HO_EFF - 1 - CROP
        w_last = WO_EFF - 1 - CROP
        if h_last < CROP or w_last < CROP:
            raise ValueError(
                f"crop<{CROP},{HO_EFF},{WO_EFF},{CO}> leaves an empty output feature map"
            )
        output_count = (h_last * WO_EFF + w_last) * CF + CF
        M = output_count * N

        problems = []
        if K not in _CALIBRATED_K:
            problems.append(f"K={K} outside calibrated set {_CALIBRATED_K}")
        if S not in _CALIBRATED_S:
            problems.append(f"S={S} not covered by calibration (only S=1 was measured)")
        if P not in _CALIBRATED_P:
            problems.append(f"P={P} outside calibrated set {_CALIBRATED_P}")
        if not (min(_CALIBRATED_H) <= H <= max(_CALIBRATED_H)) or not (
            min(_CALIBRATED_H) <= W <= max(_CALIBRATED_H)
        ):
            problems.append(f"H/W={H}/{W} outside calibrated range {_CALIBRATED_H}")
        if CI not in _CALIBRATED_CI or CO not in _CALIBRATED_CO:
            problems.append(
                f"CI={CI}/CO={CO} were never varied during calibration (always CI=CO=3)"
            )

        # The fitted overhead models deconv_swg's window-buffer fill/stall
        # behaviour, and was only ever observed in the S==1 regime, where it is
        # a large positive correction (up to ~2.2x M). Extrapolating it to
        # strided deconv is not merely inaccurate but wrong in sign: the one
        # S=2 measurement available (K=4,S=2,P=1,H=W=8,CI=2,CO=3,PE=SIMD=1,
        # FINN rtlsim = 7350) sits 30 cycles ABOVE M, i.e. the real correction
        # there is the dataflow drain latency, not thousands of stall cycles.
        # So apply the fit only inside the stride it was calibrated for.
        if S in _CALIBRATED_S:
            c = _OVERHEAD_COEFFS
            overhead = (
                c["const"]
                + c["H_EFF"] * H_EFF
                + c["SF"] * SF
                + c["K"] * K
                + c["K_H_EFF"] * K * H_EFF
                + c["K_SF"] * K * SF
            )
        else:
            overhead = 0.0
            problems.append(
                f"S={S}: swg stall overhead not modelled (fit is S=1-only); "
                f"returning the compute-bound term alone, which will slightly under-predict"
            )

        total_cycles = round(M + overhead)

        if problems:
            node_name = getattr(
                getattr(self, "onnx_node", None), "name", self.__class__.__name__
            )
            warnings.warn(
                f"{node_name}: get_exp_cycles() is extrapolating beyond its calibration data: "
                + "; ".join(problems)
            )

        return total_cycles
    # Jude: Done


    def bram_estimation(self):
        return 0

    def lut_estimation(self):
        return 0

    def uram_estimation(self):
        return 0

    def execute_node(self, context, graph):
        pass
