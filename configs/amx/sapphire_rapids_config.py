## --- Import things we need ---
import argparse
from pathlib import Path

import m5
import m5.debug

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.memory.single_channel import DIMM_DDR5_4400

# Simulation components
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

from sapphire_rapids_core import SapphireRapidsCore
from SPRThreeLevelCacheHierarchy import (
    SPR_Private_L1_Private_L2_Shared_L3_Cache_Hierarchy,
)

## =============================================================================
## --- Take in arguments for the config  (workload & amx state debug output) ---
## =============================================================================
parser = argparse.ArgumentParser(description="Run an AMX test binary")
parser.add_argument(
    "--binary",
    type=Path,
    default=Path("configs/amx/binaries/verification/intel_manual_gem5_GEMM"),
    help="path to the AMX test binary",
)
parser.add_argument(
    "--dump-directory",
    type=Path,
    default=Path("amx_debug"),
    help="directory for AMX state dump files to use for testing",
)
parser.add_argument(
    "--stdout-file",
    type=Path,
    default=None,
    help="write the binary's stdout to this file instead of gem5's output",
)

args = parser.parse_args()

binary_path = args.binary

## ================================================
## --- set up/connect things in the simualtion ---
## ================================================
# Setup Cache and Memory
memory = DIMM_DDR5_4400("1GiB")

# Setup the processor
num_cores = 1
processor = BaseCPUProcessor(
    cores=[
        SapphireRapidsCore(
            core_id=i, dump_directory=args.dump_directory.as_posix()
        )
        for i in range(num_cores)
    ]
)

# # Show CPU progress during input initialization and packing, before AMX starts.
# for core in processor.get_cores():
#     core.get_simobject().progress_interval = "10kHz"  # Every 100 simulated us.

# Defaults to Sapphire Rapids sizes and ways (see SPRThreeLevelCacheHierarchy.py)
cache_hierarchy = SPR_Private_L1_Private_L2_Shared_L3_Cache_Hierarchy()

# Setup the board (SimpleBoard is specifically used for SE mode)
board = SimpleBoard(
    clk_freq="2.1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

## ================================================
# --- add some helpers to mark the simout ---
## ================================================
# ./[path to gem5] --debug-help gives more flag that we can use

# One core cycle in ticks (1 tick = 1 ps): 1 / 2.1 GHz = 476 ticks.
TICKS_PER_CYCLE = round(1e12 / 2.1e9)


def workbegin_handler():
    while True:
        global region_start_tick
        region_start_tick = m5.curTick()
        m5.debug.flags["AMX"].enable()
        print(
            f"\n--- Start of AMX Region --- (tick {region_start_tick})\n",
            flush=True,
        )
        yield False  # Yielding true would end the simulaltion right away


def workend_handler():
    while True:
        m5.debug.flags["AMX"].disable()
        ticks = m5.curTick() - region_start_tick
        print(
            f"\n--- END of AMX Region --- (tick {m5.curTick()}, "
            f"{ticks // TICKS_PER_CYCLE} cycles)\n",
            flush=True,
        )
        yield False

## ================================================
# --- Setup the Workload and Run Simulator---
## ================================================
board.set_se_binary_workload(
    binary=BinaryResource(local_path=binary_path.as_posix()),
    stdout_file=args.stdout_file,
)

simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.WORKBEGIN: workbegin_handler(),
        ExitEvent.WORKEND: workend_handler(),
    },
)

# Trace accelerator activity during warm-up as well as the measured region.
m5.debug.flags["AMX"].enable()
print(f"Starting SE Simulation for: {binary_path.name}", flush=True)
simulator.run()
print(simulator.get_last_exit_event_cause())
print("Simulation Finished")
