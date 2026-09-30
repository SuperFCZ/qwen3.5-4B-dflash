#ifndef D_FLASH_GROUP_QUANT_LINEAR_BUILD_CONFIG_H
#define D_FLASH_GROUP_QUANT_LINEAR_BUILD_CONFIG_H

// 0 uses the available coupled AI Cores, bounded by the number of N64 tiles.
// run_server.sh writes a configured copy into its isolated build directory;
// the tracked default and the operator's public ACLNN ABI stay unchanged.
#define DFLASH_GROUP_QUANT_CORE_LIMIT 0U

#endif
