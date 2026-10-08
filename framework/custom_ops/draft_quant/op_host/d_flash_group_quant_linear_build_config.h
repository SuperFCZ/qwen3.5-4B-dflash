#ifndef D_FLASH_GROUP_QUANT_LINEAR_BUILD_CONFIG_H
#define D_FLASH_GROUP_QUANT_LINEAR_BUILD_CONFIG_H

// 0 uses the available coupled AI Cores, bounded by the number of N64 tiles.
// run_server.sh writes a configured copy into its isolated build directory;
// the tracked default and the operator's public ACLNN ABI stay unchanged.
#define DFLASH_GROUP_QUANT_CORE_LIMIT 0U

// 0 keeps the A3 instruction sequence as a benchmark control; 1 batches
// dequantization. The conditional also permits CPU-model compilation of both.
#ifndef DFLASH_GROUP_QUANT_DEQUANT_MODE
#define DFLASH_GROUP_QUANT_DEQUANT_MODE 1U
#endif

// A3.2 is opt-in until the server correctness/performance gates pass.
// 0 is the A3.1 serial control; 1 prefetches raw inputs for down and streamed
// A1 shapes. Gate/up and the whole-K256 path keep the serial schedule.
#ifndef DFLASH_GROUP_QUANT_PIPELINE_MODE
#define DFLASH_GROUP_QUANT_PIPELINE_MODE 0U
#endif
#define DFLASH_GROUP_QUANT_PREFETCH(k, n) \
    (DFLASH_GROUP_QUANT_PIPELINE_MODE == 1U && \
     (((k) == 512U || (k) == 1024U) || ((k) == 9728U && (n) == 2560U)))

// A4.1: 0 preserves A4; 1 stages only KV M80's current FP16 A tile in UB.
#ifndef DFLASH_GROUP_QUANT_KV_M80_MODE
#define DFLASH_GROUP_QUANT_KV_M80_MODE 0U
#endif
#define DFLASH_GROUP_QUANT_STAGE_A(m, k, n) \
    (DFLASH_GROUP_QUANT_KV_M80_MODE == 1U && (m) == 80U && (k) == 2560U && (n) == 2048U)

#endif
