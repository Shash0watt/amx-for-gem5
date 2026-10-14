# Stock O3 X86 core wired straight to the L1D (no AMX, no xbar), with the same
# board clock, memory and cache sizes as sapphire_rapids_config.py. The L1D's
# Cache and CachePort debug flags are on only inside the work region.
import argparse
from pathlib import Path

import m5
import m5.debug

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import DIMM_DDR5_4400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--binary",
    type=Path,
    default=Path("configs/amx/binaries/xbar/l1d_load_test"),
)
args = parser.parse_args()

board = SimpleBoard(
    clk_freq="2.1GHz",
    processor=SimpleProcessor(cpu_type=CPUTypes.O3, num_cores=1, isa=ISA.X86),
    memory=DIMM_DDR5_4400("1GiB"),
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1d_size="64KiB", l1i_size="32KiB", l2_size="2MiB"
    ),
)
board.set_se_binary_workload(
    binary=BinaryResource(local_path=args.binary.as_posix())
)

TRACE_FLAGS = ["Cache", "CachePort"]


def workbegin_handler():
    print("\n--- Start of region ---\n", flush=True)
    for flag in TRACE_FLAGS:
        m5.debug.flags[flag].enable()
    yield False


def workend_handler():
    for flag in TRACE_FLAGS:
        m5.debug.flags[flag].disable()
    print("\n--- End of region ---\n", flush=True)
    yield False


simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.WORKBEGIN: workbegin_handler(),
        ExitEvent.WORKEND: workend_handler(),
    },
)
simulator.run()
