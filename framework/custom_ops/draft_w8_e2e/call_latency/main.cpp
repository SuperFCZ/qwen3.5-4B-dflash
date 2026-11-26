// Two real production boundaries: Target Prefill -> C64, then Verify/commit -> C16.
#include "capture.hpp"
#include "qwen35_dflash/chunk.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using latency::check;
using latency::Sample;
using qwen35::dflash::AclChunkExecutor;
using Tokens = std::vector<int64_t>;
namespace fs = std::filesystem;

namespace {
Tokens read_ids(std::ifstream& in) {
    std::string line;
    check(static_cast<bool>(std::getline(in, line)), "missing workload row");
    std::istringstream row(line);
    Tokens v;
    int64_t id;
    while (row >> id) v.push_back(id);
    check(row.eof() && !v.empty(), "invalid workload row");
    return v;
}
struct Calls {
    Sample first, full, bare;
    explicit Calls(bool mark) : first(mark), full(mark), bare(mark) {}
};
void write_row(std::ostream& f, int rep, int gear, const char* name, const Sample& s) {
    f << rep << ',' << gear << ',' << name << ',' << s.model << ',' << s.wall << ',' << s.native_call;
    for (double t : s.host_ms) f << ',' << t;
    f << ',' << (s.ncopy ? s.interval(0,1) : -1) << ',' << s.interval(1,2) << ','
      << (s.ncopy ? s.interval(2,3) : -1) << '\n';
}
Tokens trial(AclChunkExecutor& executor, Calls& set, int64_t anchor, int gear, int rep,
             std::ostream& csv, const fs::path& out, bool save, bool clean, bool verify) {
    const auto role = "c" + std::to_string(gear);
    check(latency::preceding_model() == (gear == 64 ? "prefill" : "verify"), "unexpected preceding model");
    if (!clean) set.first.begin();
    const auto ids = executor.Propose(anchor, 15);
    const auto native = executor.stage_ms().at("draft").back();
    set.first.native_call = native;
    if (clean) {
        set.first.wall = native;
        write_row(csv, rep, gear, "native", set.first);
        return ids;
    }
    set.first.finish(true, role);
    std::map<std::string,std::string> inputs, outputs;
    if (verify) {
        inputs = set.first.snapshot(true, save ? out / role / "inputs" : fs::path{});
        outputs = set.first.snapshot(false, save ? out / role / "golden" : fs::path{});
        std::ofstream hashes(out / (role + "-hashes.tsv"));
        hashes.exceptions(std::ios::failbit | std::ios::badbit);
        for (const auto& item : inputs) hashes << "I " << item.first << ' ' << item.second << '\n';
        for (const auto& item : outputs) hashes << "O " << item.first << ' ' << item.second << '\n';
    }
    auto repeat = [&](Sample& hot, bool copy) {
        hot.repeat(set.first, copy);
        if (verify) {
            check(hot.snapshot(true) == inputs && hot.snapshot(false) == outputs,
                  "input mutation or replay difference in complete IDs/KV: " + role);
        }
    };
    // Replays use the captured physical datasets even though Propose has already
    // published its next KV bank. Neither replay swaps it or advances the cursor.
    const bool full_first = rep % 2 == 0;
    if (full_first) { repeat(set.full, true); repeat(set.bare, false); }
    else { repeat(set.bare, false); repeat(set.full, true); }
    auto* reported = static_cast<const int64_t*>(set.first.transfers.back().dst);
    check(ids.size() == 15 && std::equal(ids.begin(), ids.end(), reported), "hot/reported IDs differ");
    write_row(csv, rep, gear, "after_target", set.first);
    write_row(csv, rep, gear, full_first ? "hot_full" : "hot_bare", full_first ? set.full : set.bare);
    write_row(csv, rep, gear, full_first ? "hot_bare" : "hot_full", full_first ? set.bare : set.full);
    return ids;
}
Tokens proposals(Tokens ids) {
    auto eos = std::find(ids.begin(), ids.end(), 248044);
    if (eos != ids.end()) ids.erase(eos + 1, ids.end());
    return ids;
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 9, "usage: probe PLAN ROLES CASE OUT clean|pair|events POLICY REPS save|check");
        const fs::path out = argv[4];
        const std::string mode = argv[5], policy = argv[6];
        const int repetitions = std::stoi(argv[7]);
        const bool save = std::string(argv[8]) == "save";
        check(mode == "clean" || mode == "pair" || mode == "events", "invalid measurement mode");
        check(policy == "stock" || policy == "huge-io-work" || policy == "draft-weights" ||
              policy == "draft-work", "invalid allocation policy");
        check(repetitions >= 2 && repetitions <= 100 && (save || std::string(argv[8]) == "check"),
              "invalid repetitions or capture mode");
        std::ifstream workload(argv[3]);
        const auto prompt = read_ids(workload), anchor = read_ids(workload), first = read_ids(workload),
                   verified = read_ids(workload), second = read_ids(workload);
        check(prompt.size() > 16 && prompt.size() <= 64 && anchor.size() == 1 &&
              verified.size() == first.size() + 1, "invalid archive boundary");
        latency::configure(argv[2], policy);
        AclChunkExecutor executor(argv[1], 0, "dflash", true);
        std::ofstream csv(out / "samples.csv");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << std::setprecision(12)
            << "rep,gear,sequence,acl_model_id,wall_ms,native_call_ms,host_upload_ms,host_submit_ms,"
               "host_download_ms,host_sync_ms,event_input_ms,event_model_ms,event_output_ms\n";
        for (int rep = -3; rep < repetitions; ++rep) {
            // Event creation and per-iteration validation are outside the measured intervals.
            Calls c64(mode == "events"), c16(mode == "events");
            executor.Reset(0);
            const auto a = executor.Prefill(prompt, true);
            check(a == anchor[0], "Prefill anchor differs from archived real prompt");
            auto p = proposals(trial(executor, c64, a, 64, rep, csv, out, save, mode == "clean", rep == -3));
            check(p == first, "first C64 proposals differ from archive");
            Tokens block{a};
            block.insert(block.end(), p.begin(), p.end());
            const auto v = executor.Verify(block);
            check(v.size() >= verified.size() && std::equal(verified.begin(), verified.end(), v.begin()),
                  "Verify predictions differ from archive");
            size_t accepted = 0;
            while (accepted < p.size() && p[accepted] == v[accepted]) ++accepted;
            executor.Commit(accepted + 1);
            auto q = proposals(trial(executor, c16, v[accepted], 16, rep, csv, out, save, mode == "clean",
                                     rep == -3));
            check(q == second, "C16 proposals after Verify differ from archive");
        }
        executor.Close();
        csv.close();
        std::ofstream status(out / "backend.txt");
        status.exceptions(std::ios::badbit | std::ios::failbit);
#ifdef PROBE_FAKE_ACL
        status << "HOST_FIXTURE_ONLY\n";
        std::cout << "HOST_FIXTURE_ONLY\n";
#else
        status << "ASCENDCL_DEVICE_BOUNDARIES_VERIFIED\n";
        std::cout << "ASCENDCL_DEVICE_BOUNDARIES_VERIFIED\n";
#endif
        status.close();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PROBE FAILED: " << e.what() << '\n';
        return 1;
    }
}
