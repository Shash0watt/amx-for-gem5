#ifndef __AMX_ACCL_HH__
#define __AMX_ACCL_HH__

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>

#include "arch/generic/mmu.hh"
#include "cpu/base.hh"
#include "cpu/thread_context.hh"
#include "mem/packet.hh" // required for packetptr usage
#include "mem/packet_queue.hh"
#include "mem/port.hh"  // required for requestport definition
#include "mem/qport.hh" // required for queued ports
#include "params/AmxAccl.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class BaseCPU;

class AmxAccl : public ClockedObject
{

  public:
    // tunable parameters for simulation:
    int maxAmxRowLoadsPerCycle = 2;       // (not connected yet)
    int maxOutstandingCacheRequests = 16; // (not connected yet)
    int instructionQueueCapacity = 32;    // (not connected yet)

    // might also need:
    int numLoadPorts = 3;          // (not connected yet)
    int numStorePorts = 2;         // (not connected yet)
    int l1ReadBytesPerCycle = 128; // (not connected yet)

    // these are the architectural limits of palette one.
    static constexpr int MAX_ROWS = 16;
    static constexpr int MAX_COLS_BYTES = 64;
    static constexpr int NUM_TILES = 8;
    static constexpr size_t TILE_CONFIG_BYTES = 64;

    // this port sends translated reads to the cache hierarchy.
    class AmxRequestPort : public QueuedRequestPort
    {
      private:
        AmxAccl &owner;
        ReqPacketQueue reqQueue;
        SnoopRespPacketQueue snoopRespQueue;

      public:
        AmxRequestPort(const std::string &name, AmxAccl &owner);

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
    };

    AmxRequestPort memSidePort;

    Port &getPort( // override to connect python and c++
        const std::string &if_name, PortID idx = InvalidPortID) override;
    enum class MemoryTarget
    {
        TILE_ROW,
        TILE_CONFIG
    };

    struct MemoryReadChunk
    {
        MemoryTarget target;
        uint8_t tile;
        uint8_t row;
        size_t sourceOffset;
        size_t destOffset;
        size_t bytesToCopy;
    };

    // this state tells a response where its bytes belong.
    struct AmxSenderState : public Packet::SenderState
    {
        uint64_t instId;
        MemoryReadChunk readChunk;

        AmxSenderState(uint64_t inst_id, const MemoryReadChunk &read_chunk)
            : instId(inst_id), readChunk(read_chunk)
        {}
    };

    enum class AmxOpcode
    {
        AMX_CONFIG,
        AMX_LOAD,
        AMX_COMPUTE,
        AMX_STORE
    };

    struct AmxInst
    {
        uint64_t instId;  // unique id for instruction tracking
        AmxOpcode opcode; // the "type" tag
        int8_t destTile;  // destination tile index (0-7)
        int8_t srcTile1;  // source tile 1
        int8_t srcTile2;  // source tile 2

        uint64_t addr; // memory address (for loads/stores)
        size_t stride; // and also the stride

        // track the asynchronous translation and cache access stages
        uint32_t outstandingTranslations;
        uint32_t outstandingRequests;

        // prevent immediate callbacks from finishing during dispatch
        bool translationDispatchComplete;
        bool completionScheduled;

        std::array<uint8_t, TILE_CONFIG_BYTES> configData;
        Tick issueTick;

        ThreadContext *tc; // pointer to thread context

        enum class Failure
        {
            NONE,
            TRANSLATION,
            MEMORY_ERROR,
            MISSING_DATA,
            INVALID_CONFIG
        } failure;
        Fault fault;
        std::string failureReason;

        // state tracking for the scheduler
        enum class State
        {
            PENDING,
            EXECUTING,
            COMPLETED
        } state;

        // amx inst constructor
        AmxInst(uint64_t id, AmxOpcode op, int8_t dest, int8_t t1, int8_t t2,
                uint64_t addr = 0, size_t stride = 0,
                ThreadContext *_tc = nullptr)
            : instId(id),
              opcode(op),
              destTile(dest),
              srcTile1(t1),
              srcTile2(t2),
              addr(addr),
              stride(stride),
              outstandingTranslations(0),
              outstandingRequests(0),
              translationDispatchComplete(false),
              completionScheduled(false),
              configData{},
              issueTick(0),
              tc(_tc),
              failure(Failure::NONE),
              fault(NoFault),
              state(State::PENDING)
        {}
    };

    struct TileCfg
    {
        uint8_t palette_id;
        uint8_t start_row;
        uint8_t reserved_0[14];
        uint16_t colsb[NUM_TILES];
        uint8_t reserved_1[16];
        uint8_t rows[NUM_TILES];
        uint8_t reserved_2[8];
    };

    static_assert(sizeof(TileCfg) == TILE_CONFIG_BYTES);
    static_assert(offsetof(TileCfg, colsb) == 16);
    static_assert(offsetof(TileCfg, reserved_1) == 32);
    static_assert(offsetof(TileCfg, rows) == 48);
    static_assert(offsetof(TileCfg, reserved_2) == 56);

    struct TileReg
    {
        int8_t data[MAX_ROWS][MAX_COLS_BYTES];
    };

    AmxAccl(const AmxAcclParams &p);
    void startup() override;

    // this stores the parent cpu used for translation and memory requests.
    void setCPU(BaseCPU *_cpu);
    BaseCPU *
    getCPU() const
    { return cpu; }

    // todo: replace serialized pseudo-ops with a retire-aware instruction
    // path.
    void startAmxLoad(ThreadContext *tc, uint64_t dest_tile, uint64_t src_mem,
                      uint64_t stride);
    void startAmxLoadConfig(ThreadContext *tc, uint64_t config_addr);

    void tryIssue();

    // this routes returned bytes to a tile row or staged configuration.
    void handleMemResponse(PacketPtr pkt);

    void printInt8Tile(uint8_t tile_idx);
    void printInt32Tile(uint8_t tile_idx);

  private:
    // create one callback for each virtual cache-line request
    // finish() may run immediately (a tlb hit or an x86 se-mode miss) or later
    // after a page walk; only finish() receives the physical address
    // todo: model dtlb latency, translation bandwidth, outstanding limits,
    // and cpu/amx contention; x86 se-mode misses currently complete
    // immediately, so page-walk traffic requires full-system mode or a
    // synthetic se-mode model
    class AmxTranslation : public BaseMMU::Translation
    {
      private:
        AmxAccl &owner;

        // carry placement metadata until a translated packet can be created
        uint64_t instId;
        MemoryReadChunk readChunk;

      public:
        AmxTranslation(AmxAccl &owner, uint64_t inst_id,
                       const MemoryReadChunk &read_chunk);

        void markDelayed() override;
        void finish(const Fault &fault, const RequestPtr &req,
                    ThreadContext *tc, BaseMMU::Mode mode) override;
    };

    AmxInst *findReadyInstruction();
    AmxInst *findInstruction(uint64_t inst_id);
    bool allTilesIdle() const;
    void executeInstruction(AmxInst *ready_inst);
    void executeLoadInstruction(AmxInst *inst);
    void executeConfigInstruction(AmxInst *inst);
    void dispatchMemoryReadChunk(AmxInst *inst, uint64_t virtual_addr,
                                 size_t request_size,
                                 const MemoryReadChunk &read_chunk);
    void finishTranslation(uint64_t inst_id,
                           const MemoryReadChunk &read_chunk,
                           const Fault &fault, const RequestPtr &req);
    void maybeFinishMemoryInstruction(uint64_t inst_id);
    void finishLoadInstruction(AmxInst *inst);
    void finishConfigInstruction(uint64_t inst_id);
    TileCfg decodeTileConfig(const std::array<uint8_t, TILE_CONFIG_BYTES> &raw)
        const;
    bool validateTileConfig(const TileCfg &config, std::string &reason) const;
    void clearTiles();
    void eraseInstruction(uint64_t inst_id);

    // keep the parent cpu association used by the amx instruction path
    BaseCPU *cpu;

    // internal registers for amx.
    // todo: make amx state per-thread before supporting smt.
    TileCfg currentCfg;       // global configuration register
    TileReg tiles[NUM_TILES]; // matrix register file
    bool tilesConfigured;
    const Cycles configLatency;
    EventFunctionWrapper configCompletionEvent;
    uint64_t configCompletionInstId;
    bool issuingInstructions = false;

    // for out of order logic
    std::deque<AmxInst> instructionQueue;
    struct ScoreBoardEntry
    {
        int readerCount = 0;
        bool writeActive = false;
    };
    ScoreBoardEntry tileScoreboard[NUM_TILES];
    uint64_t nextInstId = 0; // counter to assign unique ids
};

} // namespace gem5

#endif // __amx_accl_hh__
