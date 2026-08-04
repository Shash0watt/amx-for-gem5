#include "amx/instruction_amx.hh"

namespace gem5
{

namespace amx
{

Instruction::Instruction(uint64_t id, Opcode opcode, int8_t destination,
                         int8_t source1, int8_t source2, uint64_t address,
                         size_t stride, ThreadContext *tc)
    : id(id),
      opcode(opcode),
      destination(destination),
      source1(source1),
      source2(source2),
      address(address),
      stride(stride),
      threadContext(tc)
{}

Instruction
Instruction::tileLoad(uint64_t id, ThreadContext *tc, uint8_t destination,
                      uint64_t address, size_t stride)
{
    return Instruction(id, Opcode::Load, destination, -1, -1, address, stride,
                       tc);
}

Instruction
Instruction::tileConfig(uint64_t id, ThreadContext *tc, uint64_t address)
{
    return Instruction(id, Opcode::Config, -1, -1, -1, address, 0, tc);
}

Instruction
Instruction::tileDotProduct(uint64_t id, uint8_t destination,
                            uint8_t source1, uint8_t source2)
{
    return Instruction(id, Opcode::DotProduct, destination, source1, source2,
                       0, 0, nullptr);
}

Instruction
Instruction::tileZero(uint64_t id, uint8_t destination)
{
    return Instruction(id, Opcode::Zero, destination, -1, -1, 0, 0,
                       nullptr);
}

bool
Instruction::readsTile(int tile) const
{
    if (opcode == Opcode::DotProduct) {
        return source1 == tile || source2 == tile || destination == tile;
    }

    return opcode == Opcode::Store && source1 == tile;
}

bool
Instruction::writesTile(int tile) const
{
    return (opcode == Opcode::Load || opcode == Opcode::DotProduct ||
            opcode == Opcode::Zero) && destination == tile;
}

bool
Instruction::hasRAW(bool younger_reads, bool older_writes)
{
    return younger_reads && older_writes;
}

bool
Instruction::hasWAR(bool younger_writes, bool older_reads)
{
    return younger_writes && older_reads;
}

bool
Instruction::hasWAW(bool younger_writes, bool older_writes)
{
    return younger_writes && older_writes;
}

bool
Instruction::hasRAW(const Instruction &older) const
{
    for (int tile = 0; tile < NumTiles; ++tile) {
        if (hasRAW(readsTile(tile), older.writesTile(tile))) {
            return true;
        }
    }

    return false;
}

bool
Instruction::hasWAR(const Instruction &older) const
{
    for (int tile = 0; tile < NumTiles; ++tile) {
        if (hasWAR(writesTile(tile), older.readsTile(tile))) {
            return true;
        }
    }

    return false;
}

bool
Instruction::hasWAW(const Instruction &older) const
{
    for (int tile = 0; tile < NumTiles; ++tile) {
        if (hasWAW(writesTile(tile), older.writesTile(tile))) {
            return true;
        }
    }

    return false;
}

} // namespace amx
} // namespace gem5
