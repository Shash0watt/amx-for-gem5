#ifndef __AMX_INSTRUCTION_AMX_HH__
#define __AMX_INSTRUCTION_AMX_HH__

#include <cstddef>
#include <cstdint>
#include <string>

#include "amx/tile_amx.hh"
#include "base/types.hh"
#include "sim/faults.hh"

namespace gem5
{

class ThreadContext;

namespace amx
{

enum class Opcode
{
    Config,
    Load,
    DotProduct,
    Zero,
    Store
};

struct Instruction
{
    enum class Failure
    {
        None,
        Translation,
        MemoryError,
        MissingData,
        InvalidConfig
    };

    enum class State
    {
        Pending,
        Executing,
        Completed
    };

    static Instruction tileLoad(uint64_t id, ThreadContext *tc,
                                uint8_t destination, uint64_t address,
                                size_t stride);
    static Instruction tileStore(uint64_t id, ThreadContext *tc,
                                 uint8_t source, uint64_t address,
                                 size_t stride);
    static Instruction tileConfig(uint64_t id, ThreadContext *tc,
                                  uint64_t address);

    static Instruction tileDotProduct(uint64_t id, uint8_t destination,
                                      uint8_t source1, uint8_t source2);
    static Instruction tileZero(uint64_t id, uint8_t destination);

    bool readsTile(int tile) const;
    bool writesTile(int tile) const;

    // Test accesses to one tile. The first access belongs to the younger
    // instruction and the second belongs to the older instruction.
    static bool hasRAW(bool younger_reads, bool older_writes);
    static bool hasWAR(bool younger_writes, bool older_reads);
    static bool hasWAW(bool younger_writes, bool older_writes);

    // Test all tiles used by this instruction against an older instruction.
    bool hasRAW(const Instruction &older) const;
    bool hasWAR(const Instruction &older) const;
    bool hasWAW(const Instruction &older) const;

    uint64_t id;
    Opcode opcode;
    int8_t destination;
    int8_t source1;
    int8_t source2;

    uint64_t address;
    size_t stride;
    ThreadContext *threadContext;

    uint32_t outstandingTranslations = 0;
    uint32_t outstandingRequests = 0;
    bool translationDispatchComplete = false;
    bool completionScheduled = false;
    bool latencyElapsed = false;
    bool memoryComplete = false;

    RawTileConfig configData = {};
    Tick issueTick = 0;

    Failure failure = Failure::None;
    Fault fault = NoFault;
    std::string failureReason;
    State state = State::Pending;

  private:
    Instruction(uint64_t id, Opcode opcode, int8_t destination, int8_t source1,
                int8_t source2, uint64_t address, size_t stride,
                ThreadContext *tc);
};

} // namespace amx
} // namespace gem5

#endif // __AMX_INSTRUCTION_AMX_HH__
