#include "amx/amx_instruction.hh"

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

bool
Instruction::readsTile(int tile) const
{
    if (opcode == Opcode::Compute) {
        return source1 == tile || source2 == tile || destination == tile;
    }

    return opcode == Opcode::Store && source1 == tile;
}

bool
Instruction::writesTile(int tile) const
{
    return (opcode == Opcode::Load || opcode == Opcode::Compute) &&
           destination == tile;
}

bool
Instruction::hasTileHazardWith(const Instruction &older) const
{
    for (int tile = 0; tile < NumTiles; ++tile) {
        if ((readsTile(tile) && older.writesTile(tile)) ||
            (writesTile(tile) &&
             (older.readsTile(tile) || older.writesTile(tile)))) {
            return true;
        }
    }

    return false;
}

} // namespace amx
} // namespace gem5
