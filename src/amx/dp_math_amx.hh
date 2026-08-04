#ifndef __AMX_DP_MATH_AMX_HH__
#define __AMX_DP_MATH_AMX_HH__

#include <cstdint>
#include <cstring>

#include "amx/tile_amx.hh"
#include "base/logging.hh"

namespace gem5
{
namespace amx
{

namespace detail
{

inline float
readBFloat16(const int8_t *bytes)
{
    uint16_t bfloat_bits = 0;
    std::memcpy(&bfloat_bits, bytes, sizeof(bfloat_bits));

    const uint32_t float_bits = static_cast<uint32_t>(bfloat_bits) << 16;
    float value = 0.0F;
    std::memcpy(&value, &float_bits, sizeof(value));
    return value;
}

} // namespace details

// Check that the three tile operands form a valid BF16 matrix operation.
inline void
validateDotProductOp(const TileConfig &config, int destination, int source1,
                     int source2)
{
    panic_if(destination < 0 || destination >= NumTiles || source1 < 0 ||
                 source1 >= NumTiles || source2 < 0 || source2 >= NumTiles,
             "AMX dot product has an invalid tile operand");
    panic_if(destination == source1 || destination == source2 ||
                 source1 == source2,
             "AMX TDPBF16PS requires three distinct tile registers");

    const uint16_t destination_rows = config.rows[destination];
    const uint16_t destination_colsb = config.columnBytes[destination];
    const uint16_t source1_rows = config.rows[source1];
    const uint16_t source1_colsb = config.columnBytes[source1];
    const uint16_t source2_rows = config.rows[source2];
    const uint16_t source2_colsb = config.columnBytes[source2];

    panic_if(destination_rows == 0 || destination_colsb == 0 ||
                 source1_rows == 0 || source1_colsb == 0 ||
                 source2_rows == 0 || source2_colsb == 0,
             "AMX TDPBF16PS requires three configured tiles");
    panic_if(destination_rows != source1_rows,
             "AMX TDPBF16PS destination and source-1 row counts differ");
    panic_if(destination_colsb != source2_colsb,
             "AMX TDPBF16PS destination and source-2 column sizes differ");
    panic_if(source1_colsb != source2_rows * sizeof(uint32_t),
             "AMX TDPBF16PS source-1 columns do not match source-2 rows");
    panic_if(destination_colsb % sizeof(float) != 0,
             "AMX TDPBF16PS destination column size is not FP32-aligned");
}

// Multiply BF16 pairs and accumulate their FP32 results into the destination.
inline void
doDotProductBF16(const TileConfig &config, TileRegisterFile &tiles,
                 int destination, int source1, int source2)
{
    const uint8_t rows = config.rows[destination];
    const uint8_t inner_groups = config.rows[source2];
    const uint16_t output_columns =
        config.columnBytes[destination] / sizeof(float);

    for (uint8_t row = 0; row < rows; ++row) {
        for (uint16_t column = 0; column < output_columns; ++column) {
            int8_t *destination_value =
                &tiles[destination].data[row][column * sizeof(float)];
            float accumulator = 0.0F;
            std::memcpy(&accumulator, destination_value, sizeof(accumulator));

            for (uint8_t group = 0; group < inner_groups; ++group) {
                const int8_t *source1_pair =
                    &tiles[source1].data[row][group * sizeof(uint32_t)];
                const int8_t *source2_pair =
                    &tiles[source2].data[group][column * sizeof(uint32_t)];

                accumulator +=
                    detail::readBFloat16(source1_pair) *
                        detail::readBFloat16(source2_pair) +
                    detail::readBFloat16(source1_pair + sizeof(uint16_t)) *
                        detail::readBFloat16(source2_pair + sizeof(uint16_t));
            }

            std::memcpy(destination_value, &accumulator, sizeof(accumulator));
        }
    }
}

} // namespace amx
} // namespace gem5

#endif // __AMX_DP_MATH_AMX_HH__
