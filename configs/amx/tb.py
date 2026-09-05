import argparse
from pathlib import Path

import m5
import m5.debug

from m5.objects import AmxAccl

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.memory.single_channel import DIMM_DDR5_4400

# Simulation components
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

# Make the AMX-specific cache hierarchy available to this testbench.
m5.util.addToPath("../../src/amx/amxXbar")

from amx_private_l1_private_l2_cache_hierarchy import (
    AmxPrivateL1PrivateL2CacheHierarchy,
)

parser = argparse.ArgumentParser(description="Run one asynchronous AMX test")
parser.add_argument(
    "--binary",
    type=Path,
    default=Path("configs/amx/binaries/simple_mm"),
    help="path to the AMX test binary",
)
parser.add_argument(
    "--dump-directory",
    type=Path,
    default=Path("amx_debug"),
    help="directory for AMX state dump files",
)
args = parser.parse_args()

binary_path = args.binary


# Setup Cache and Memory
memory = DIMM_DDR5_4400("1GiB")

# Setup the processor
# (CPUTypes.ATOMIC is faster for purely functional tests, but TIMING is better if you need cycle counts)
processor = SimpleProcessor(
    # cpu_type=CPUTypes.TIMING,  # in order proc
    cpu_type=CPUTypes.O3,  # config for Out of Order
    num_cores=1,
    isa=ISA.X86,
)

# attach the AMX Accelerator to the CPU(s)
# the SimpleProcessor wraps the actual CPU SimObjects.
# we iterate through the cores and attach our accelerator
# directly to the underlying BaseCPU (core.core).
for core in processor.cores:
    core.core.amx_accl = AmxAccl(
        dump_directory=args.dump_directory.as_posix()
    )

    # comment out if not out of order
    ## TODO: Check these numbers with actaul documentation
    ## TODO: Make sure that execution units, etc other CPU parameters are accurate to saphire rapids
    
    core.core.decodeWidth = 6
    core.core.renameWidth = 8
    core.core.dispatchWidth = 8
    core.core.issueWidth = 8
    core.core.commitWidth = 8

    core.core.numROBEntries = 512
    core.core.LQEntries = 192
    core.core.SQEntries = 114

    core.core.numPhysIntRegs = 280
    core.core.numPhysFloatRegs = 332

cache_hierarchy = AmxPrivateL1PrivateL2CacheHierarchy(
    l1d_size="48KiB",
    l1i_size="32KiB",
    l2_size="2MiB",
)

# Setup the board (SimpleBoard is specifically used for SE mode)
board = SimpleBoard(
    clk_freq="2.9GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# Setup Workload
board.set_se_binary_workload(
    binary=BinaryResource(local_path=binary_path.as_posix())
)

# ./[path to gem5] --debug-help gives more flag that we can use


start_tick = 0




def workbegin_handler():
    print(f"\n--- Start of AMX ROI (Tick: {start_tick}, Cycle: {start_cycle}) ---\n")

    m5.debug.flags["AMX"].enable()

    yield False  # Yielding False tells the simulator to continue running


def workend_handler():
    end_tick = m5.curTick()
    clk_period = get_clk_period_ticks()
    start_cycle = int(start_tick / clk_period)
    end_cycle = int(end_tick / clk_period)
    print(f"\n--- Start of AMX ROI (Tick: {start_tick}, Cycle: {start_cycle}) ---\n")

    elapsed_ticks = end_tick - start_tick
    elapsed_cycles = end_cycle - start_cycle
    yield False


# Setup and Run Simulator
simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.WORKBEGIN: workbegin_handler(),
        ExitEvent.WORKEND: workend_handler(),
    },
)

print(f"Starting SE Simulation for: {binary_path.name}")
simulator.run()
print("Simulation Done")
