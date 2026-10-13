#pragma once
// Generated copies only are changed by run_server.sh. 0 = available AI Cores.
#define DFLASH_HEAD_CORE_LIMIT 0U

// Experimental only; server builds select explicitly. Default is the device-accepted compare_select conversion.
// 0=cast, 1=minmax, 2=compare_select
#define DFLASH_HEAD_SATURATION 2

// 0=accepted Scalar Top1, 1=experimental Vector Top1.
#define DFLASH_HEAD_VECTOR_TOP1 0
// 0=correctness kernel; 1=Matmul observer; 2=Matmul+conversion observer.
#define DFLASH_HEAD_PROFILE_STAGE 0
