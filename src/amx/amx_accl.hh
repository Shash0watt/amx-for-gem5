#ifndef __AMX_ACCL_HH__
#define __AMX_ACCL_HH__

#include "cpu/base.hh"
#include "cpu/thread_context.hh"
#include "mem/packet.hh" // Required for PacketPtr usage
#include "mem/packet_queue.hh"
#include "mem/port.hh"  // Required for RequestPort definition
#include "mem/qport.hh" // Required for using Queued Ports
#include "params/AmxAccl.hh"
#include "sim/clocked_object.hh"

#include <deque>

namespace gem5
{

class BaseCPU;

class AmxAccl : public ClockedObject
{

  public:
    // tunable parameters for simulation:
    int maxAmxRowLoadsPerCycle = 2; // (not connected yet)
    int maxOutstandingCacheRequests = 16; // (not connected yet)
    int instructionQueueCapacity = 32; // (not connected yet)

    // might also need:
    int numLoadPorts = 3; // (not connected yet)
    int numStorePorts = 2; // (not connected yet)
    int l1ReadBytesPerCycle = 128; // (not connected yet)

    // Memory Port Class
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

    Port &getPort( // overrride to connect python & c++
        const std::string &if_name,
        PortID idx = InvalidPortID) override;
    // used for async port req tracking with packets
    // we use this to identify exactly which tile and row the payload belongs
    // to
    struct AmxSenderState : public Packet::SenderState
    {
        uint64_t instId;     // unique id of instruction that generated request
        uint8_t destTile;    // Target TMM tile register (0-7)
        uint8_t rowIdx;      // Which matrix row this fragment belongs to
        uint8_t cacheOffset; // Where the data starts within the returned 64B
                             // cache line packet
        uint16_t rowOffset;  // Where this fragment should be inserted into the
                             // internal tile matrix row
        size_t bytesToCopy;  // The explicit number of bytes to extract from
                             // this specific sub-packet
        AmxSenderState(uint64_t inst_id, uint8_t dest, uint8_t row, uint8_t c_offset,
                       uint16_t r_offset, size_t bytes)
            : instId(inst_id),
              destTile(dest),
              rowIdx(row),
              cacheOffset(c_offset),
              rowOffset(r_offset),
              bytesToCopy(bytes)
        {}
    };

    enum class AmxOpcode
    {
        AMX_LOAD,
        AMX_COMPUTE,
        AMX_STORE
    };

    struct AmxInst
    {
        uint64_t instId;  // unique ID for instruction tracking
        AmxOpcode opcode; // the "type" tag
        int8_t destTile;  // destination tile index (0-7)
        int8_t srcTile1;  // Source tile 1
        int8_t srcTile2;  // Source tile 2 (for when we are computing values)

        uint64_t addr; // memory address (for loads/stores)
        size_t stride; // and also the stride

        uint32_t outstandingRequests; // the conuter for memory responses
        ThreadContext *tc;            // pointer to thread context

        enum class Failure
        {
            NONE,
            TRANSLATION,
            MEMORY_ERROR,
            MISSING_DATA
        } failure;
        Fault fault;

        // state tracking for the scheduler
        enum class State
        {
            PENDING,
            EXECUTING,
            COMPLETED
        } state;

        // amx inst constructor
        AmxInst(uint64_t id, AmxOpcode op, int8_t dest, int8_t t1, int8_t t2,
                uint64_t addr = 0, uint32_t stride = 0, ThreadContext *_tc = nullptr)
            : instId(id),
              opcode(op),
              destTile(dest),
              srcTile1(t1),
              srcTile2(t2),
              addr(addr),
              stride(stride),
              outstandingRequests(0),
              tc(_tc),
              failure(Failure::NONE),
              fault(NoFault),
              state(State::PENDING)
        {}
    };

    static constexpr int MAX_ROWS = 16;
    static constexpr int MAX_COLS_BYTES = 64;
    static constexpr int NUM_TILES = 8;

    struct TileCfg
    {
        uint8_t palette_id;
        uint8_t start_row;
        uint8_t reserved_0[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    };

    struct TileReg
    {
        uint16_t rows;
        uint16_t colbytes;
        int8_t data[MAX_ROWS][MAX_COLS_BYTES];
    };

    AmxAccl(const AmxAcclParams &p);
    void startup() override;

    // Sets the parent CPU reference for accessing its memory ports.
    void setCPU(BaseCPU *_cpu);
    BaseCPU *
    getCPU() const
    { return cpu; }

    void startAmxLoad(ThreadContext *tc, uint64_t dest_tile, uint64_t src_mem,
                      uint64_t stride);

    void tryIssue();

    // Handles memory responses routed from the CPU. Replaces
    // AmxMemPort::recvTimingResp.
    void handleMemResponse(PacketPtr pkt);

    void printInt8Tile(uint8_t tile_idx);
    void printInt32Tile(uint8_t tile_idx);

  private:
    AmxInst *findReadyInstruction();
    void executeInstruction(AmxInst *ready_inst);
    void finishLoadInstruction(AmxInst *inst);

    // Pointer to the parent CPU. In core multiplexing, we access the cache
    // pipeline directly through the CPU's data port, removing the need for a
    // separate RequestPort.
    BaseCPU *cpu;

    // internal registers for AMX.
    TileCfg currentCfg;       // Global config state register
    TileReg tiles[NUM_TILES]; // Matrix register file (TMM0 - TMM7)

    // keeps track of exactly how many sub-requests remain outstanding for each
    // tile. when tileOutstandingRequests[tile_idx] reaches 0, the tile load is
    // complete.
    size_t tileOutstandingRequests[NUM_TILES]; // TODO: replace this with the
                                               // AmxInst object

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

#endif // __AMX_ACCL_HH__
