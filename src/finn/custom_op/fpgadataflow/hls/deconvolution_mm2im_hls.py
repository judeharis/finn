# Jude: Created MM2IMv2
import numpy as np
import os

from finn.custom_op.fpgadataflow.deconvolution_mm2im import DeconvolutionMM2IM
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.basic import roundup_to_integer_multiple
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)


class DeconvolutionMM2IM_hls(DeconvolutionMM2IM, HLSBackend):
    """Corresponds to finn-hlslib mm2im() (internal_embedded) and mm2im_stream()
    (internal_decoupled, external). One call processes a whole frame, so it uses the
    default (ifm_aware) cppsim template, not the free-running timeout loop.

    Streamed weights: input 1 carries one PE*SIMD word per compute step, element
    pe*SIMD + simd (element 0 in the LSBs, hls::vector compact=bit), the same calc_wmem()
    words for every input pixel. internal_decoupled wraps the HLS IP and a memstream
    looping over those words in one block-design hierarchy, as Thresholding_hls does."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(DeconvolutionMM2IM.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """[OFM][K][K][IFM] -> (1, WMEM, PE, SIMD) in mm2im's order
        t = ((ky*K + kx)*CF + cf)*SF + sf, co = cf*PE + pe, ci = sf*SIMD + simd."""
        k_h, k_w = self.get_nodeattr("KernelDim")
        ofm_ch, ifm_ch = self.get_nodeattr("OFMChannels"), self.get_nodeattr("IFMChannels")
        pe, simd = self.get_nodeattr("PE"), self.get_nodeattr("SIMD")
        cf, sf = self._cf_sf()
        assert orig_weight_matrix.shape == (ofm_ch, k_h, k_w, ifm_ch), (
            "Weights matrix doesn't have expected shape (ofm_ch, k_h, k_w, ifm_ch)"
        )
        ret = orig_weight_matrix.reshape(cf, pe, k_h, k_w, sf, simd)
        ret = ret.transpose(2, 3, 0, 4, 1, 5)
        return ret.reshape(1, self.calc_wmem(), pe, simd)

    def get_weight_stream_words(self, weights):
        """One pixel's weight stream: (WMEM, PE*SIMD), element pe*SIMD + simd."""
        pe, simd = self.get_nodeattr("PE"), self.get_nodeattr("SIMD")
        return self.get_hw_compatible_weight_tensor(weights).reshape(-1, pe * simd)

    def generate_params(self, model, path):
        weights = model.get_initializer(self.onnx_node.input[1])
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            self.make_weight_file(weights, "hls_header", "{}/params.h".format(path))
        elif mem_mode == "internal_decoupled":
            # memstream contents (ipgen); cppsim gets input_1.npy from execute_node
            self.make_weight_file(weights, "decoupled_verilog_dat", "{}/memblock.dat".format(path))

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Write the weights as an HLS ROM header (hls_header), one pixel's weight stream
        as .npy (decoupled_npy) or as memstream hex lines (decoupled_verilog_dat)."""
        export_wdt = self.get_weight_datatype()
        if weight_file_mode == "hls_header":
            weight_tensor = self.get_hw_compatible_weight_tensor(weights)
            weight_hls_code = numpy_to_hls_code(weight_tensor, export_wdt, "weights", False, True)
            # remove framing {}
            weight_hls_code = weight_hls_code[1:-2] + ";"
            with open(weight_file_name, "w") as f:
                f.write(
                    "static {} const weights[{}][{}][{}] = ".format(
                        export_wdt.get_hls_datatype_str(),
                        self.calc_wmem(),
                        self.get_nodeattr("PE"),
                        self.get_nodeattr("SIMD"),
                    )
                )
                f.write(weight_hls_code)
        elif weight_file_mode == "decoupled_npy":
            np.save(weight_file_name, self.get_weight_stream_words(weights).astype(np.float32))
        elif weight_file_mode == "decoupled_verilog_dat":
            # element 0 in the LSBs, as hls::vector compact=bit
            width = roundup_to_integer_multiple(self.get_instream_width(1), 4)
            words = pack_innermost_dim_as_hex_string(
                self.get_weight_stream_words(weights), export_wdt, width,
                reverse_inner=True, prefix="",
            )
            with open(weight_file_name, "w") as f:
                for word in words.flatten():
                    f.write(word + "\n")
        else:
            raise Exception("Unknown weight_file_mode %s" % weight_file_mode)

    def fold_input_for_npy(self, inp_val, ind):
        # cppsim's input_1.npy holds one pixel's weight words; read_npy_data replays them
        if ind == 1 and self.streams_weights():
            return self.get_weight_stream_words(inp_val)
        return super().fold_input_for_npy(inp_val, ind)

    def read_npy_data(self):
        super().read_npy_data()
        if not self.streams_weights():
            return
        # the generic code reads input_1.npy with input 0's vector width; replace it by a
        # read of the PE*SIMD-wide words, replayed for every input pixel
        reads = self.code_gen_dict["$READNPYDATA$"]
        self.code_gen_dict["$READNPYDATA$"] = [r for r in reads if "in1_V" not in r]
        npy_in = "%s/input_1.npy" % self.get_nodeattr("code_gen_dir_cppsim")
        wtype = self.get_weight_datatype().get_hls_datatype_str()
        width = self.get_nodeattr("PE") * self.get_nodeattr("SIMD")
        vec = "hls::vector<%s, %d>" % (wtype, width)
        self.code_gen_dict["$READNPYDATA$"] += [
            "{",
            'hls::stream<%s> wpix("wpix");' % vec,
            'npy2vectorstream<%s, float, %d>("%s", wpix, false);' % (wtype, width, npy_in),
            "std::vector<%s> wseq;" % vec,
            "while(!wpix.empty()) wseq.push_back(wpix.read());",
            "for(unsigned pix = 0; pix < IFMH*IFMW; pix++) for(auto const &w : wseq) in1_V.write(w);",
            "}",
        ]

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "mm2im.hpp"']

    def defines(self, var):
        k, s, p, h, w = self._geom()
        self.code_gen_dict["$DEFINES$"] = [
            "constexpr unsigned Kernel = %d;" % k,
            "constexpr unsigned Stride = %d;" % s,
            "constexpr unsigned Padding = %d;" % p,
            "constexpr unsigned IFMH = %d;" % h,
            "constexpr unsigned IFMW = %d;" % w,
            "constexpr unsigned ICH = %d;" % self.get_nodeattr("IFMChannels"),
            "constexpr unsigned OCH = %d;" % self.get_nodeattr("OFMChannels"),
            "constexpr unsigned SIMD1 = %d;" % self.get_nodeattr("SIMD"),
            "constexpr unsigned PE1 = %d;" % self.get_nodeattr("PE"),
            "constexpr bool SKIP1 = %s;" % ("true" if self.get_nodeattr("SKIP") else "false"),
            "using TW = %s;" % self.get_weight_datatype().get_hls_datatype_str(),
            "using TI = %s;" % self.get_input_datatype().get_hls_datatype_str(),
            "using TO = %s;" % self.get_output_datatype().get_hls_datatype_str(),
            "using TA = %s;" % self.get_accumulator_datatype().get_hls_datatype_str(),
        ]

    def docompute(self):
        if self.streams_weights():
            call = (
                "mm2im_stream<Kernel, Stride, Padding, IFMH, IFMW, OCH, ICH, PE1, SIMD1, "
                "TW, TI, TO, TA>(in1_V, in0_V, out0_V);"
            )
        else:
            call = (
                "mm2im<Kernel, Stride, Padding, IFMH, IFMW, OCH, ICH, PE1, SIMD1, "
                "TW, TI, TO, TA, SKIP1>(weights, in0_V, out0_V);"
            )
        self.code_gen_dict["$DOCOMPUTE$"] = [call]

    def blackboxfunction(self):
        streams = [
            "hls::stream<hls::vector<%s, %d>> &in0_V"
            % (self.get_input_datatype().get_hls_datatype_str(), self.get_nodeattr("SIMD"))
        ]
        if self.streams_weights():
            streams.append(
                "hls::stream<hls::vector<%s, %d>> &in1_V"
                % (
                    self.get_weight_datatype().get_hls_datatype_str(),
                    self.get_nodeattr("PE") * self.get_nodeattr("SIMD"),
                )
            )
        streams.append(
            "hls::stream<hls::vector<%s, %d>> &out0_V"
            % (self.get_output_datatype().get_hls_datatype_str(), self.get_nodeattr("PE"))
        )
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(%s)" % (self.onnx_node.name, ", ".join(streams))
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            # sub-byte hls::vector elements packed to the width FINN expects (see rev2d)
            "#pragma HLS aggregate variable=in0_V compact=bit",
            "#pragma HLS aggregate variable=out0_V compact=bit",
        ]
        if self.streams_weights():
            self.code_gen_dict["$PRAGMAS$"] += [
                "#pragma HLS INTERFACE axis port=in1_V",
                "#pragma HLS aggregate variable=in1_V compact=bit",
            ]
            return
        self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
        ram_style = self.get_nodeattr("ram_style")
        if ram_style != "auto":
            impl = {"block": "bram", "distributed": "lutram"}[ram_style]
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS bind_storage variable=weights type=rom_1p impl=%s" % impl
            )

    def execute_node(self, context, graph):
        if not (self.streams_weights() and self.get_nodeattr("exec_mode") == "rtlsim"):
            # cppsim replays input_1.npy per pixel itself (read_npy_data)
            HLSBackend.execute_node(self, context, graph)
            return
        # rtlsim with streamed weights: feed the weight sequence once per input pixel
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        x = context[node.input[0]].astype(np.float32)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), self.fold_input_for_npy(x, 0))
        inp = npy_to_rtlsim_input(
            os.path.join(code_gen_dir, "input_0.npy"),
            self.get_input_datatype(0),
            self.get_instream_width(0),
        )
        w_path = os.path.join(code_gen_dir, "input_1.npy")
        self.make_weight_file(context[node.input[1]], "decoupled_npy", w_path)
        wei = npy_to_rtlsim_input(w_path, self.get_weight_datatype(), self.get_instream_width(1))
        h, w = self.get_nodeattr("IFMDim")
        io_dict = {"inputs": {"in0": inp, "in1": wei * (h * w)}, "outputs": {"out0": []}}
        sim = self.get_rtlsim()
        self.reset_rtlsim(sim)
        self.rtlsim_multi_io(sim, io_dict)
        self.close_rtlsim(sim)
        out_npy_path = os.path.join(code_gen_dir, "output_0.npy")
        odt = self.get_output_datatype()
        rtlsim_output_to_npy(
            io_dict["outputs"]["out0"], out_npy_path, odt, self.get_folded_output_shape(),
            self.get_outstream_width(), odt.bitwidth(),
        )
        context[node.output[0]] = (
            np.load(out_npy_path).astype(np.float32).reshape(self.get_normal_output_shape())
        )

    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape(0)[:-1])
        io_dict = {"inputs": {"in0": [0 for i in range(n_inps)]}, "outputs": {"out0": []}}
        if self.streams_weights():
            n_w = np.prod(self.get_folded_input_shape(1)[:-1])
            io_dict["inputs"]["in1"] = [0 for i in range(n_w)]
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)

    def code_generation_ipgen(self, model, fpgapart, clk):
        super().code_generation_ipgen(model, fpgapart, clk)
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.generate_hdl_memstream(fpgapart)

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            # in1_V is driven by the memstream inside the layer's hierarchy
            intf_names["s_axis"] = [i for i in intf_names["s_axis"] if i[0] != "in1_V"]
        return intf_names

    def code_generation_ipi(self):
        if self.get_nodeattr("mem_mode") != "internal_decoupled":
            return super().code_generation_ipi()
        # a hierarchy with the layer's port names, holding the HLS IP and its memstream
        node_name = self.onnx_node.name
        intf = self.get_verilog_top_module_intf_names()
        clk_name, rst_name = intf["clk"][0], intf["rst"][0]
        din_name, dout_name = intf["s_axis"][0][0], intf["m_axis"][0][0]
        cmd = [
            "create_bd_cell -type hier %s" % node_name,
            "create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name),
            "create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name),
            "create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s"
            % (node_name, dout_name),
            "create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s"
            % (node_name, din_name),
            "create_bd_cell -type ip -vlnv %s /%s/%s"
            % (self.get_nodeattr("ip_vlnv"), node_name, node_name),
        ]
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
        ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
        strm_tmpl = node_name + "_memstream_wrapper.v"
        for f in [
            os.path.join(code_gen_dir, strm_tmpl),
            axi_dir + "axilite.sv",
            ms_rtllib_dir + "memstream_axi.sv",
            ms_rtllib_dir + "memstream.sv",
        ]:
            cmd.append("add_files -norecurse %s" % f)
        strm_inst = node_name + "_wstrm"
        cmd += [
            "create_bd_cell -type hier -reference %s /%s/%s"
            % (strm_tmpl[:-2], node_name, strm_inst),
            "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
            "[get_bd_intf_pins %s/%s/in1_V]" % (node_name, strm_inst, node_name, node_name),
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
            % (node_name, rst_name, node_name, strm_inst),
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
            % (node_name, clk_name, node_name, strm_inst),
            # no pumped memory: the 2x clock input takes the 1x clock
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
            % (node_name, clk_name, node_name, strm_inst),
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
            % (node_name, rst_name, node_name, node_name, rst_name),
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
            % (node_name, clk_name, node_name, node_name, clk_name),
            "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s/%s]"
            % (node_name, din_name, node_name, node_name, din_name),
            "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s/%s]"
            % (node_name, dout_name, node_name, node_name, dout_name),
            "save_bd_design",
        ]
        return cmd
