# Jude: Created MM2IMv2
import math
import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.deconvolution import Deconvolution


class DeconvolutionMM2IM(Deconvolution):
    """Transposed convolution by input-stationary scatter (MM2IM), finn-hlslib mm2im.hpp.

    Same interface, weight tensor [OFM][K][K][IFM] and numpy execution as Deconvolution
    (rev2d, deconv.hpp), with three differences:

    * any square K, S and P < K, including K % S != 0 and K < S (rev2d needs S | K);
    * the products are accumulated in accDataType, which MinimizeAccumulatorWidth narrows
      from the weight values (the accumulator array is the layer's main memory);
    * SKIP=1 visits only the kernel taps whose output survives the padding crop.

    mem_mode selects where the weights live, as for the MVAU:

    * internal_embedded: a ROM inside the HLS kernel (mm2im()).
    * internal_decoupled: a memstream next to the kernel streams them (mm2im_stream()).
    * external: streamed from outside the layer through input 1.

    Streamed weights are one PE*SIMD word per compute step, the same K*K*CF*SF words for
    every input pixel, so SKIP is ignored (off) unless the weights are embedded.
    """

    # Cycle model: loop iterations + CYCLE_ROW[SKIP] per input row + CYCLE_CONST, the
    # pipeline fill/flush of the compute and drain loops (shallower without SKIP). Fitted to
    # 12 Vitis HLS 2024.1 cosim runs and the 58 rtlsim runs of the MM2IM sweep test; accurate
    # to max(3%, 6 cycles per input row). Same constants as finn-hlslib tb/gen_mm2im_golden.py.
    CYCLE_ROW = {1: 21, 0: 17}
    CYCLE_CONST = -34

    def get_nodeattr_types(self):
        my_attrs = {
            "accDataType": ("s", False, "INT32"),
            # skip the taps cropped away; internal_embedded only (see get_skip)
            "SKIP": ("i", False, 1, {0, 1}),
            "mem_mode": (
                "s",
                False,
                "internal_embedded",
                {"internal_embedded", "internal_decoupled", "external"},
            ),
            # weight memory: the HLS ROM (internal_embedded; auto lets HLS decide, BRAM
            # above about 1 kbit per lane) or the memstream (internal_decoupled)
            "ram_style": ("s", False, "auto", {"auto", "block", "distributed"}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_accumulator_datatype(self):
        return DataType[self.get_nodeattr("accDataType")]

    def get_input_datatype(self, ind=0):
        if ind == 1:
            return self.get_weight_datatype()
        return super().get_input_datatype(ind)

    def streams_weights(self):
        return self.get_nodeattr("mem_mode") in ["internal_decoupled", "external"]

    def get_skip(self):
        """SKIP as implemented: streamed weights need every tap, so it is off for them."""
        return 0 if self.streams_weights() else self.get_nodeattr("SKIP")

    def calc_wmem(self):
        """Weight words per PE x SIMD lane, one per compute step of a pixel: K*K*CF*SF."""
        cf, sf = self._cf_sf()
        return int(np.prod(self.get_nodeattr("KernelDim")) * cf * sf)

    def get_instream_width(self, ind=0):
        if ind == 1:
            if not self.streams_weights():
                return 0
            pe, simd = self.get_nodeattr("PE"), self.get_nodeattr("SIMD")
            return pe * simd * self.get_weight_datatype().bitwidth()
        return super().get_instream_width(ind)

    def get_folded_input_shape(self, ind=0):
        if ind == 1 and self.streams_weights():
            # every input pixel replays the whole weight sequence
            h, w = self.get_nodeattr("IFMDim")
            pe, simd = self.get_nodeattr("PE"), self.get_nodeattr("SIMD")
            return (1, h * w * self.calc_wmem(), pe * simd)
        return super().get_folded_input_shape(ind)

    # -- geometry (mirrors finn-hlslib mm2im_sched.hpp) ---------------------------------
    def _geom(self):
        k_h, k_w = self.get_nodeattr("KernelDim")
        s_h, s_w = self.get_nodeattr("Stride")
        p_h, p_w = self.get_nodeattr("Padding")
        if k_h != k_w or s_h != s_w or p_h != p_w:
            raise ValueError("mm2im needs a square KernelDim, Stride and Padding")
        h, w = self.get_nodeattr("IFMDim")
        return k_h, s_h, p_h, h, w

    def _cf_sf(self):
        ofm_ch, ifm_ch = self.get_nodeattr("OFMChannels"), self.get_nodeattr("IFMChannels")
        pe, simd = self.get_nodeattr("PE"), self.get_nodeattr("SIMD")
        if ofm_ch % pe != 0 or ifm_ch % simd != 0:
            raise ValueError("PE must divide OFMChannels and SIMD must divide IFMChannels")
        return ofm_ch // pe, ifm_ch // simd

    @staticmethod
    def _tap_range(i, k, s, p, o):
        """Taps [lo, hi) of input index i whose output i*s + tap lands in [p, p + o)."""
        lo = 0 if i * s >= p else p - i * s
        hi = 0 if p + o <= i * s else min(k, p + o - i * s)
        return lo, hi

    def get_iterations(self):
        """Compute + drain loop iterations of one frame (mm2im_geom::iterations)."""
        k, s, p, h, w = self._geom()
        cf, sf = self._cf_sf()
        skip = self.get_skip()
        ho, wo = (h - 1) * s + k - 2 * p, (w - 1) * s + k - 2 * p
        bands = h + (k - 1) // s

        def taps(i, o):
            lo, hi = self._tap_range(i, k, s, p, o)
            return hi - lo if skip else k

        taps_x = sum(taps(i, wo) for i in range(w))
        n = sum(taps(i, ho) * taps_x * cf * sf for i in range(h))
        for b in range(bands):
            row_lo = 0 if b * s >= p else min(s, p - b * s)
            row_hi = 0 if p + ho <= b * s else min(s, p + ho - b * s)
            n += max(0, row_hi - row_lo) * wo * cf
        return n

    def get_exp_cycles(self):
        h = self.get_nodeattr("IFMDim")[0]
        row = self.CYCLE_ROW[self.get_skip()]
        return int(self.get_iterations() + max(0, row * h + self.CYCLE_CONST))

    def _acc_depth(self):
        """Accumulator entries per PE lane: KB bands x S rows x WO columns x CF."""
        k, s, p, h, w = self._geom()
        cf, _ = self._cf_sf()
        wo = (w - 1) * s + k - 2 * p
        return ((k - 1) // s + 1) * s * wo * cf

    def verify_node(self):
        info = []
        try:
            k, s, p, h, w = self._geom()
            self._cf_sf()
        except ValueError as e:
            return [str(e)]
        if p >= k:
            info.append("mm2im needs Padding < KernelDim")
        if 2 * p >= (min(h, w) - 1) * s + k:
            info.append("Padding crops the whole output")
        if self.streams_weights() and self.get_nodeattr("SKIP"):
            info.append("SKIP is ignored with streamed weights (mem_mode %s)"
                        % self.get_nodeattr("mem_mode"))
        return info

    # -- accumulator width ----------------------------------------------------------
    def _acc_range(self, weights, idt):
        """Bounds on every accumulator value. Output position o = i*S + k only receives
        taps k = o (mod S), so outputs split into S*S phases; per output channel and phase
        the product ranges [min(w*xlo, w*xhi), max(...)] are summed (FINN's
        calculate_matvec_accumulator_range per phase). The input range contains 0, so every
        partial sum held in the accumulator is within the bound too."""
        s = self.get_nodeattr("Stride")[0]
        xlo, xhi = idt.min(), idt.max()
        w = np.asarray(weights, dtype=np.float64)
        lo = hi = 0
        for ry in range(s):
            for rx in range(s):
                wp = w[:, ry::s, rx::s, :].reshape(w.shape[0], -1)
                if wp.shape[1] == 0:
                    continue
                lo = min(lo, np.minimum(wp * xlo, wp * xhi).sum(axis=1).min())
                hi = max(hi, np.maximum(wp * xlo, wp * xhi).sum(axis=1).max())
        return int(lo), int(hi)

    def minimize_accumulator_width(self, model, datatype_only=False):
        """Narrowest accDataType for the weight values (or, with datatype_only, external
        weights or no initializer, for the weight datatype bounds); the output datatype
        follows it, as for an MVAU without activation."""
        weights = model.get_initializer(self.onnx_node.input[1])
        idt = self.get_input_datatype()
        if datatype_only or weights is None or self.get_nodeattr("mem_mode") == "external":
            wdt = self.get_weight_datatype()
            shape = self.get_normal_input_shape(1)
            r = [self._acc_range(np.full(shape, v), idt) for v in (wdt.min(), wdt.max())]
            acc_min, acc_max = min(r[0][0], r[1][0]), max(r[0][1], r[1][1])
        else:
            acc_min, acc_max = self._acc_range(weights, idt)
        if acc_min >= 0:
            adt = DataType["UINT%d" % max(1, math.ceil(math.log2(acc_max + 1)))]
        else:
            adt = DataType["INT%d" % (math.ceil(math.log2(max(-acc_min, 1 + acc_max))) + 1)]
        odt = adt
        # a graph output is byte-aligned, as for the MVAU
        if model.find_direct_successors(self.onnx_node) is None:
            bw = roundup_to_integer_multiple(adt.bitwidth(), 8)
            odt = DataType[adt.name.replace(str(adt.bitwidth()), str(bw))]
        self.set_nodeattr("accDataType", adt.name)
        self.set_nodeattr("outputDataType", odt.name)
        model.set_tensor_datatype(self.onnx_node.output[0], odt)
        return adt

    # -- resources (provisional; to be calibrated against OOC synthesis) ---------------
    @staticmethod
    def _bram18(depth, width):
        """BRAM18 for one depth x width memory, 0 below about 1 kbit (LUTRAM in practice).
        Accumulator: true dual port, so at most 18 bits wide per BRAM18."""
        if depth * width <= 1024:
            return 0
        return math.ceil(width / 18) * math.ceil(depth / 1024)

    def _weight_rom_bram18(self):
        """BRAM18 of the weight memory: PE*SIMD ROM lanes of WMEM x wbits inside the kernel
        (internal_embedded), one WMEM x PE*SIMD*wbits memstream (internal_decoupled), or
        none (external)."""
        style = self.get_nodeattr("ram_style")
        mem_mode = self.get_nodeattr("mem_mode")
        if style == "distributed" or mem_mode == "external":
            return 0
        depth, width = self.calc_wmem(), self.get_weight_datatype().bitwidth()
        lanes = self.get_nodeattr("PE") * self.get_nodeattr("SIMD")
        if mem_mode == "internal_decoupled":
            width, lanes = width * lanes, 1
        if style == "auto" and depth * width <= 1024:
            return 0
        # single-port ROM aspect ratios 512x36, 1Kx18, 2Kx9, 4Kx4, 8Kx2, 16Kx1: the widest
        # one deep enough
        per = next((b for d, b in [(512, 36), (1024, 18), (2048, 9), (4096, 4), (8192, 2)]
                    if depth <= d), 1)
        return lanes * math.ceil(width / per) * max(1, math.ceil(depth / 16384))

    def bram_estimation(self, fpgapart):
        acc = self.get_nodeattr("PE") * self._bram18(
            self._acc_depth(), self.get_accumulator_datatype().bitwidth()
        )
        return acc + self._weight_rom_bram18()

    def lut_estimation(self, fpgapart):
        # control + per-multiplier datapath (as rev2d) + weight ROM in LUTs (64 bits/LUT6)
        macs = self.get_nodeattr("PE") * self.get_nodeattr("SIMD")
        lut = 850 + 30 * macs
        if self.get_nodeattr("mem_mode") != "external" and self._weight_rom_bram18() == 0:
            k = self.get_nodeattr("KernelDim")[0]
            lut += (k * k * self.get_nodeattr("IFMChannels") * self.get_nodeattr("OFMChannels")
                    * self.get_weight_datatype().bitwidth()) / 64
        return int(lut)

    def dsp_estimation(self, fpgapart):
        return self.get_nodeattr("PE") * self.get_nodeattr("SIMD")

    def uram_estimation(self, fpgapart):
        return 0
