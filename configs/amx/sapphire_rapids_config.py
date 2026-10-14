## --- Import things we need ---
import argparse
from pathlib import Path

import m5
import m5.debug

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import DIMM_DDR5_4400

# Simulation components
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

from sapphire_rapids_core import SapphireRapidsCore

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

cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    # Sapphire Rapids has a 48 KiB 12-way L1D, but the stock hierarchy is fixed at
    # 8-way and gem5 needs a power-of-two set count, so I'm rounding up to 64 KiB
    l1d_size="64KiB",
    l1i_size="32KiB",
    l2_size="2MiB",
)

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
def workbegin_handler():
    m5.debug.flags["AMX"].enable()
    print(f"\n--- Start of AMX Region ---\n")
    yield False  # Yielding true would end the simulaltion right away


def workend_handler():
    m5.debug.flags["AMX"].disable()
    print(f"\n--- END of AMX Region ---\n")
    yield False 

## ================================================
# --- Setup the Workload and Run Simulator---
## ================================================
board.set_se_binary_workload(
    binary=BinaryResource(local_path=binary_path.as_posix())
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
