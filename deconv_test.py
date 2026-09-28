# Jude: Created
from tests.fpgadataflow.test_fpgadataflow_deconv import test_fpgadataflow_deconv_revd2


idim = [8, 8]
stride = [2, 2]
ifm_ch = 2
ofm_ch = 3
simd = 1
pe = 1
k = 4
padding = 1


# idim = [128,128]
# stride = [2, 2]
# ifm_ch = 32
# ofm_ch = 3
# simd = 1
# pe = 1
# k = 6
# padding = 2

# idim = [2,2]
# stride = [2, 2]
# ifm_ch = 32
# ofm_ch = 3
# simd = 1
# pe = 1
# k = 6
# padding = 2

# exec_mode = ["cppsim", "rtlsim"]
exec_mode = "rtlsim"


test_fpgadataflow_deconv_revd2(
    idim, stride, ifm_ch, ofm_ch, simd, pe, k, padding, exec_mode
)
