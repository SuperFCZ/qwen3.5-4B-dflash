// Device-function compile check only; not an executable Kernel or NPU result.
#include "kernel_tpipe.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_block_sync_intf.h"
using namespace AscendC;
__aicore__ void checked_copy(GlobalTensor<half> out, LocalTensor<half> src) {
    DataCopy(out, src, DataCopyParams{16, 1, 15, 0});
    PipeBarrier<PIPE_ALL>();
}
