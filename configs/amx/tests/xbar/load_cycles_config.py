# usage:
#   ./build/X86/gem5.opt configs/amx/tests/xbar/load_cycles_config.py --cpu xbar
#   ./build/X86/gem5.opt configs/amx/tests/xbar/load_cycles_config.py --cpu no-xbar
#   ./build/X86/gem5.opt configs/amx/tests/xbar/load_cycles_config.py --cpu sapphire-rapids
import argparse
import sys
from pathlib import Path

import m5
import m5.debug

from xbar_cpu import XbarCPU

# SapphireRapidsCore lives two directories up, in configs/amx.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from sapphire_rapids_core import SapphireRapidsCore

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import (
    PrivateL1CacheHierarchy,
)
from gem5.components.memory.single_channel import DIMM_DDR5_4400
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cpu", choices=["xbar", "no-xbar", "sapphire-rapids"], default="xbar"
)
parser.add_argument(
    "--binary",
    type=Path,
    default=Path("configs/amx/binaries/xbar/load_cycles_test"),
)
args = parser.parse_args()

CLK_FREQ_GHZ = 2.1

if args.cpu == "xbar":
    processor = BaseCPUProcessor(
        cores=[XbarCPU(core_id=0, dump_directory="amx_debug")]
    )
elif args.cpu == "sapphire-rapids":
    processor = BaseCPUProcessor(
        cores=[SapphireRapidsCore(core_id=0, dump_directory="amx_debug")]
    )
else:
    processor = SimpleProcessor(cpu_type=CPUTypes.O3, num_cores=1, isa=ISA.X86)

board = SimpleBoard(
    clk_freq=f"{CLK_FREQ_GHZ}GHz",
    processor=processor,
    memory=DIMM_DDR5_4400("1GiB"),
    cache_hierarchy=PrivateL1CacheHierarchy(l1d_size="64KiB", l1i_size="32KiB"),
)
board.set_se_binary_workload(
    binary=BinaryResource(local_path=args.binary.as_posix())
)

# LSQUnit:      the CPU sends a load to the L1D ("Memory request ... was sent").
#               The packet already prints as ReadResp: the line is logged after
#               the send returns, and the L1D turns the request into its
#               response in place.
# Cache:        the L1D receives it and looks it up ("access for ReadReq ... hit").
# PacketQueue:  the L1D schedules the response back ("schedSendTiming for
#               ReadResp ... when <tick>"; <tick> is when the CPU gets it).
TRACE_FLAGS = ["Cache"]

# Hide the L1I's lines so the trace is just the L1D's. Each L1D line is one
# request arriving; lines with the same tick arrived in the same cycle.
m5.trace.ignore("board.cache_hierarchy.l1icaches")


def workbegin_handler():
    print("\n--- Start of region (1 cycle = 476 ticks) ---\n", flush=True)
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