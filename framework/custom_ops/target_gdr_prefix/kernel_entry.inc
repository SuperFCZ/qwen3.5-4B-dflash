extern "C" __global__ __aicore__ void target_gdr_prefix_state(
    GM_ADDR key, GM_ADDR value, GM_ADDR g, GM_ADDR beta, GM_ADDR initial,
    GM_ADDR length, GM_ADDR state, GM_ADDR workspace, GM_ADDR tiling) {
    using namespace AscendC;
    GET_TILING_DATA(t, tiling);
    GlobalTensor<int16_t> count;
    count.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(length), 1);
    const int16_t r = count.GetValue(0);
    if (r < 1 || r > 16) return;
    PrefixNativeConfig config{t.pruneLevel};
    TPipe pipe;
    GM_ADDR user = GetUserWorkspace(workspace);
    Qwen3Next::ChunkGatedDeltaRuleKernel<Qwen3Next::GatedDeltaRuleMode::CHUNK> kernel;
    // The old helper has seven inputs. Its Query has no data dependence on
    // the committed state. The unpruned clone tests this alias on the board.
    kernel.Init(key, key, value, g, beta, initial, length, user + 32768, state, user, config, &pipe);
    kernel.Process();
}
