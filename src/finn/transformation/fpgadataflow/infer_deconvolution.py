# Jude: Created
import numpy as np
import warnings
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class InferDeconvolution(Transformation):
    """
    Convert ConvTranspose (NCHW) nodes into a fused Deconvolution HW layer (NHWC)
    surrounded by Transpose nodes, as an alternative to InferPixelPaddingDeconv.
    Pixel padding computes the transposed convolution as an ordinary convolution
    over a zero-stuffed image, so for stride S only 1/S^2 of its MACs touch real
    data; the fused kernel skips the zeros.

    Only ConvTranspose nodes the finn-hlslib deconv kernel can implement are
    converted: group 1, dilation 1, no bias, square kernel/stride/padding, K
    divisible by S (rev2d only), P < K (mm2im only), integer input and weights, and an output that fits the INT32
    accumulator. Anything else is left in place (with a warning) for
    InferPixelPaddingDeconv. PE and SIMD start at 1 and are set by folding.

    The Deconvolution output is the raw INT32 accumulator; a following
    MultiThreshold is not absorbed and becomes a standalone Thresholding layer.
    The Transpose nodes need to be streamlined away afterwards, e.g. with
    AbsorbConsecutiveTransposes and AbsorbTransposeIntoMultiThreshold.

    impl selects the HW layer:

    * "rev2d" (default): Deconvolution, finn-hlslib deconv.hpp (gather; needs S | K).
    * "mm2im": DeconvolutionMM2IM, finn-hlslib mm2im.hpp (input-stationary scatter).
      Also takes K % S != 0 and K < S (any P < K). accDataType starts at INT32;
      MinimizeAccumulatorWidth narrows it (and the output) from the weight values.
      mem_mode (mm2im only) is internal_embedded (weight ROM, SKIP on),
      internal_decoupled (memstream) or external (weights streamed in); the streamed
      modes visit every tap (SKIP off).
    """

    def __init__(self, impl="rev2d", mem_mode="internal_embedded"):
        super().__init__()
        assert impl in ("rev2d", "mm2im"), "impl must be rev2d or mm2im, got %s" % impl
        assert impl == "mm2im" or mem_mode == "internal_embedded", "rev2d embeds its weights"
        self.impl = impl
        self.mem_mode = mem_mode

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in list(graph.node):
            if n.op_type != "ConvTranspose":
                continue
            params = self._match(model, n)
            if params is None:
                continue
            (k, s, p, ifm_ch, ofm_ch, ifm_dim, ofm_dim, idt, wdt, odt, W) = params
            deconv_input = n.input[0]
            deconv_output = n.output[0]

            # ConvTranspose weights are [IFM][OFM][k][k]; the kernel wants
            # [OFM][k][k][IFM]. No 180 degree rotation: unlike pixel padding, the
            # fused kernel computes the transposed convolution directly.
            w_name = model.make_new_valueinfo_name()
            model.set_initializer(w_name, W.transpose(1, 2, 3, 0))
            model.set_tensor_datatype(w_name, wdt)

            inp_trans_out = model.make_new_valueinfo_name()
            model.set_tensor_shape(
                inp_trans_out, [1, ifm_dim[0], ifm_dim[1], ifm_ch], TensorProto.FLOAT
            )
            model.set_tensor_datatype(inp_trans_out, idt)
            deconv_out = model.make_new_valueinfo_name()
            model.set_tensor_shape(
                deconv_out, [1, ofm_dim[0], ofm_dim[1], ofm_ch], TensorProto.FLOAT
            )
            model.set_tensor_datatype(deconv_out, odt)

            # NCHW -> NHWC
            inp_trans_node = helper.make_node(
                "Transpose", [deconv_input], [inp_trans_out], perm=[0, 2, 3, 1]
            )
            if self.impl == "mm2im":
                # one call per frame: the default (ifm_aware) cppsim template
                skip = int(self.mem_mode == "internal_embedded")
                op_type = "DeconvolutionMM2IM"
                impl_attrs = dict(accDataType=odt.name, SKIP=skip, mem_mode=self.mem_mode)
            else:
                op_type, impl_attrs = "Deconvolution", dict(hls_style="freerunning")
            deconv_node = helper.make_node(
                op_type,
                [inp_trans_out, w_name],
                [deconv_out],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                KernelDim=[k, k],
                IFMChannels=ifm_ch,
                OFMChannels=ofm_ch,
                IFMDim=list(ifm_dim),
                Stride=[s, s],
                Padding=[p, p],
                PE=1,
                SIMD=1,
                inputDataType=idt.name,
                weightDataType=wdt.name,
                outputDataType=odt.name,
                name=op_type + "_" + n.name,
                cpp_interface="hls_vector",
                **impl_attrs,
            )
            # NHWC -> NCHW
            out_trans_node = helper.make_node(
                "Transpose", [deconv_out], [deconv_output], perm=[0, 3, 1, 2]
            )
            model.set_tensor_datatype(deconv_output, odt)

            # insert nodes where the ConvTranspose is to preserve topological ordering
            node_ind = list(graph.node).index(n)
            graph.node.insert(node_ind, out_trans_node)
            graph.node.insert(node_ind, deconv_node)
            graph.node.insert(node_ind, inp_trans_node)
            graph.node.remove(n)
            graph_modified = True

        return (model, graph_modified)

    def _match(self, model, n):
        """Return the Deconvolution parameters for ConvTranspose node n, or None
        (after a warning) if the fused kernel can't implement it."""

        def skip(reason):
            warnings.warn("%s: %s. Can't infer Deconvolution." % (n.name, reason))
            return None

        group = get_by_name(n.attribute, "group")
        if group is not None and group.i != 1:
            return skip("only group=1 is supported")
        if len(n.input) > 2:
            return skip("bias input is not supported")
        W = model.get_initializer(n.input[1])
        if W is None:
            return skip("weights must be an initializer")
        ishape = model.get_tensor_shape(n.input[0])
        oshape = model.get_tensor_shape(n.output[0])
        if len(ishape) != 4 or ishape[0] != 1:
            return skip("only 2D ConvTranspose with batch size 1 is supported")

        dilation = get_by_name(n.attribute, "dilations")
        if dilation is not None and any(d != 1 for d in dilation.ints):
            return skip("only dilation 1 is supported")
        auto_pad = get_by_name(n.attribute, "auto_pad")
        if auto_pad is not None and auto_pad.s.decode("utf-8") != "NOTSET":
            return skip("only explicit padding is supported")
        k_attr = get_by_name(n.attribute, "kernel_shape")
        kernel = list(k_attr.ints) if k_attr is not None else list(W.shape[2:])
        s_attr = get_by_name(n.attribute, "strides")
        stride = list(s_attr.ints) if s_attr is not None else [1, 1]
        p_attr = get_by_name(n.attribute, "pads")
        pads = list(p_attr.ints) if p_attr is not None else [0, 0, 0, 0]
        # the HLS kernel takes a single Kernel/Stride/Padding for both dimensions
        if len(set(kernel)) != 1 or len(set(stride)) != 1 or len(set(pads)) != 1:
            return skip("only square kernel, stride and symmetric padding are supported")
        k, s, p = kernel[0], stride[0], pads[0]
        if self.impl == "rev2d" and k % s != 0:
            return skip("kernel size %d is not divisible by stride %d" % (k, s))
        if self.impl == "mm2im" and p >= k:
            return skip("padding %d is not smaller than the kernel size %d" % (p, k))

        ifm_ch, ifm_dim = ishape[1], (ishape[2], ishape[3])
        ofm_ch = W.shape[1]
        ofm_dim = tuple((d - 1) * s - 2 * p + k for d in ifm_dim)
        # also catches output_padding / output_shape, which the kernel can't do
        if list(oshape) != [1, ofm_ch, ofm_dim[0], ofm_dim[1]]:
            return skip("output shape %s is not the plain ConvTranspose shape" % str(oshape))

        idt = model.get_tensor_datatype(n.input[0])
        wdt = model.get_tensor_datatype(n.input[1])
        if not (idt.is_integer() and wdt.is_integer()):
            return skip("input and weights must be integer, got %s / %s" % (idt, wdt))

        # The kernel accumulates in its output type, so bound the accumulator:
        # at most every weight of an output channel meets an extreme input.
        x_ext = max(abs(idt.min()), abs(idt.max()))
        acc_bound = (np.abs(W).sum(axis=(0, 2, 3)) * x_ext).max()
        odt = DataType["INT32"]
        if acc_bound > odt.max():
            return skip("accumulator bound %d does not fit %s" % (acc_bound, odt.name))

        return (k, s, p, ifm_ch, ofm_ch, ifm_dim, ofm_dim, idt, wdt, odt, W)
