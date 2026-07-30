#ifndef __AMX_AMX_INSTRUCTION_HH__
#define __AMX_AMX_INSTRUCTION_HH__

#include <cstddef>
#include <cstdint>
#include <string>

#include "amx/amx_tile.hh"
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
    Compute,
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
    static Instruction tileConfig(uint64_t id, ThreadContext *tc,
                                  uint64_t address);

    bool readsTile(int tile) const;
    bool writesTile(int tile) const;
    bool hasTileHazardWith(const Instruction &older) const;

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

#endif // __AMX_AMX_INSTRUCTION_HH__
