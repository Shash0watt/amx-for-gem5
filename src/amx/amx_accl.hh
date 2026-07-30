#ifndef __AMX_AMX_ACCL_HH__
#define __AMX_AMX_ACCL_HH__

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>

#include "amx/amx_instruction.hh"
#include "amx/amx_tile.hh"
#include "arch/generic/mmu.hh"
#include "mem/packet.hh"
#include "mem/packet_queue.hh"
#include "mem/qport.hh"
#include "params/AmxAccl.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class BaseCPU;
class ThreadContext;

//  Timing model for Intel AMX in gem5! tile configuration and tile-memory
//  operations.

//  Public start methods submit architectural operations. The
//  scheduler selects hazard-free work, execution methods describe each
//  operation, and the memory pipeline handles asynchronous translation and
//  cache responses.

class AmxAccl : public ClockedObject
{
  public:
    static constexpr int MAX_ROWS = amx::MaxRows;
    static constexpr int MAX_COLS_BYTES = amx::MaxColumnBytes;
    static constexpr int NUM_TILES = amx::NumTiles;
    static constexpr size_t TILE_CONFIG_BYTES = amx::TileConfigBytes;

    explicit AmxAccl(const AmxAcclParams &params);

    void startup() override;
    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;

    void setCPU(BaseCPU *cpu);
    BaseCPU *
    getCPU() const
    { return cpu; }

    // TODO: replace serialized pseudo-ops with a retire-aware instruction
    // path.
    void startAmxLoad(ThreadContext *tc, uint64_t destination, uint64_t source,
                      uint64_t stride);
    void startAmxLoadConfig(ThreadContext *tc, uint64_t config_address);
    void tryIssue();

  private:
    using AmxInst = amx::Instruction;
    using AmxOpcode = amx::Opcode;

    // ---------------------------------------------------------------------
    // Memory port and translation callback
    // ---------------------------------------------------------------------
    class AmxRequestPort : public QueuedRequestPort
    {
      public:
        AmxRequestPort(const std::string &name, AmxAccl &owner);

      protected:
        bool recvTimingResp(PacketPtr pkt) override;

      private:
        AmxAccl &owner;
        ReqPacketQueue requestQueue;
        SnoopRespPacketQueue snoopResponseQueue;
    };

    enum class MemoryTarget
    {
        TileRow,
        TileConfig
    };

    struct MemoryReadChunk
    {
        MemoryTarget target;
        uint8_t tile;
        uint8_t row;
        size_t sourceOffset;
        size_t destinationOffset;
        size_t bytesToCopy;
    };

    struct AmxSenderState : public Packet::SenderState
    {
        AmxSenderState(uint64_t instruction_id,
                       const MemoryReadChunk &read_chunk)
            : instructionId(instruction_id), readChunk(read_chunk)
        {}

        uint64_t instructionId;
        MemoryReadChunk readChunk;
    };

    class AmxTranslation : public BaseMMU::Translation
    {
      public:
        AmxTranslation(AmxAccl &owner, uint64_t instruction_id,
                       const MemoryReadChunk &read_chunk);

        void markDelayed() override;
        void finish(const Fault &fault, const RequestPtr &request,
                    ThreadContext *tc, BaseMMU::Mode mode) override;

      private:
        AmxAccl &owner;
        uint64_t instructionId;
        MemoryReadChunk readChunk;
    };

    // ---------------------------------------------------------------------
    // Instruction scheduling and hazard detection
    // ---------------------------------------------------------------------
    AmxInst *findReadyInstruction();
    AmxInst *findInstruction(uint64_t instruction_id);
    bool hasActiveTileHazard(const AmxInst &instruction) const;
    bool hasOlderTileHazard(const AmxInst &instruction) const;
    bool allTilesIdle() const;
    void eraseInstruction(uint64_t instruction_id);

    // ---------------------------------------------------------------------
    // Instruction execution
    // ---------------------------------------------------------------------
    void executeInstruction(AmxInst *instruction);
    void executeLoadInstruction(AmxInst *instruction);
    void executeConfigInstruction(AmxInst *instruction);
    void executeComputeInstruction(AmxInst *instruction);
    void executeStoreInstruction(AmxInst *instruction);
    void finishLoadInstruction(AmxInst *instruction);
    void finishConfigInstruction(uint64_t instruction_id);
    void commitTileConfig(const amx::TileConfig &config);

    // ---------------------------------------------------------------------
    // Translation and timed memory access
    // ---------------------------------------------------------------------
    void beginMemoryInstruction(AmxInst *instruction);
    void finishMemoryDispatch(AmxInst *instruction);
    void dispatchMemoryRead(AmxInst *instruction, uint64_t virtual_address,
                            size_t bytes, MemoryTarget target,
                            uint8_t tile = 0, uint8_t row = 0);
    void dispatchMemoryReadChunk(AmxInst *instruction,
                                 uint64_t virtual_address, size_t request_size,
                                 const MemoryReadChunk &read_chunk);
    void finishTranslation(uint64_t instruction_id,
                           const MemoryReadChunk &read_chunk,
                           const Fault &fault, const RequestPtr &request);
    void handleMemoryResponse(PacketPtr packet);
    void validateMemoryReadOwner(const AmxInst &instruction,
                                 const MemoryReadChunk &read_chunk) const;
    void *memoryDestination(AmxInst &instruction,
                            const MemoryReadChunk &read_chunk);
    void maybeFinishMemoryInstruction(uint64_t instruction_id);

    // ---------------------------------------------------------------------
    // SimObject connections and architectural state
    // ---------------------------------------------------------------------
    AmxRequestPort memSidePort;
    BaseCPU *cpu;

    amx::TileConfig currentConfig;
    amx::TileRegisterFile tiles;
    bool tilesConfigured;

    const Cycles configLatency;
    EventFunctionWrapper configCompletionEvent;
    uint64_t configCompletionInstructionId;

    struct ScoreboardEntry
    {
        int readerCount = 0;
        bool writeActive = false;
    };

    std::deque<AmxInst> instructionQueue;
    std::array<ScoreboardEntry, NUM_TILES> tileScoreboard;
    uint64_t nextInstructionId = 0;
    bool issuingInstructions = false;
};

} // namespace gem5

#endif // __AMX_AMX_ACCL_HH__
