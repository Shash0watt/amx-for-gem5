#include <cstdint>

#include <gem5/m5ops.h>

typedef struct
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

static_assert(sizeof(__tilecfg) == 64);

// this workload checks that configuration controls the following tile load.
int
main()
{
    alignas(64) __tilecfg config = {};
    config.palette_id = 1;
    config.colsb[0] = 8;
    config.rows[0] = 2;
    config.colsb[1] = 8;
    config.rows[1] = 2;

    alignas(64) int32_t source_0[2][2] = {
        {1, 2},
        {3, 4},
    };
    alignas(64) int32_t source_1[2][2] = {
        {5, 6},
        {7, 8},
    };

    m5_work_begin(0, 0);
    amx_tile_loadconfig(&config);
    amx_tile_loadd(0, source_0, sizeof(source_0[0]));
    amx_tile_loadd(1, source_1, sizeof(source_1[0]));

    // the simuout should show both load issues before either load completes
    // this loadconfig should never issue before a older instrcution
    alignas(64) __tilecfg release = {};
    amx_tile_loadconfig(&release);

    // TODO: replace this wait with amx actual signaling
    m5_quiesce_cycle(
        10000); // wait for some cycles to ensure the tile loads are complete
    m5_work_end(0, 0);

    return 0;
}
