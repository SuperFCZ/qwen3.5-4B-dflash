#pragma once

namespace qwen35::dflash {
// One static OM, one FP16 input/output, four vectors. Python compares outputs.
int RunMatmulProbe(int argc, char** argv);
}
