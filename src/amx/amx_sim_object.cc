#include "amx/amx_accl.hh"
#include "base/trace.hh"
#include "cpu/base.hh"
#include "debug/AMX.hh"

namespace gem5
{

// -------------------------------------------------------------------------
// SimObject lifecycle and connections
// -------------------------------------------------------------------------

AmxAccl::AmxAccl(const AmxAcclParams &params)
    : ClockedObject(params),
      memSidePort(name() + ".mem_side", *this),
      cpu(nullptr),
      currentConfig{},
      tilesConfigured(false),
      configLatency(params.config_latency),
      loadLatency(params.load_latency),
      dotProductLatency(params.dp_latency),
      zeroLatency(params.zero_latency),
      storeLatency(params.store_latency),
      resourceTracker(params.load_issue_throughput,
                      params.dp_issue_throughput,
                      params.zero_issue_throughput,
                      params.store_issue_throughput),
      configCompletionEvent(
          [this] { processConfigCompletionEvent(); },
          name() + ".config_completion"),
      pendingConfigInstructionId(0),
      issueRetryEvent(
          [this] { processIssueRetryEvent(); }, name() + ".issue_retry"),
      latencyEvent(
          [this] { processLatencyEvent(); }, name() + ".instruction_latency")
{
    amx::clearTiles(tiles);
    DPRINTF(AMX, "Created the AMX SimObject\n");
}

void
AmxAccl::processConfigCompletionEvent()
{
    finishConfigInstruction(pendingConfigInstructionId);
}

void
AmxAccl::startup()
{
    DPRINTF(AMX, "AMX object started up\n");
}

void
AmxAccl::setCPU(BaseCPU *new_cpu)
{
    panic_if(!new_cpu, "AMX accelerator requires a valid parent CPU");

    cpu = new_cpu;
    DPRINTF(AMX, "Connected AMX accelerator to parent CPU %s\n", cpu->name());
}

Port &
AmxAccl::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "mem_side") {
        return memSidePort;
    }

    return ClockedObject::getPort(if_name, idx);
}

// -------------------------------------------------------------------------
// Queued memory-side port
// -------------------------------------------------------------------------

AmxAccl::AmxRequestPort::AmxRequestPort(const std::string &name,
                                        AmxAccl &owner)
    : QueuedRequestPort(name, requestQueue, snoopResponseQueue),
      owner(owner),
      requestQueue(owner, *this),
      snoopResponseQueue(owner, *this)
{}

bool
AmxAccl::AmxRequestPort::recvTimingResp(PacketPtr packet)
{
    owner.handleMemoryResponse(packet);
    return true;
}

} // namespace gem5
