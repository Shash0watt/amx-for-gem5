## --- Import things we need ---
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


## --- Take in arguments for the config  (workload & amx state debug output) ---
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

## --- set up/connect things in the simualtion ---
# Setup Cache and Memory
memory = DIMM_DDR5_4400("1GiB")

# Setup the processor
processor = SimpleProcessor(
    # cpu_type=CPUTypes.TIMING,  # in order proc
    cpu_type=CPUTypes.O3,  # config for Out of Order
    num_cores=1,
    isa=ISA.X86,
)

## the way the baseCPU class expects the simObjects to be attached is through passing it as a paramter
## we iterate through each core attach the amx simObject
for core in processor.cores:
    core.core.amx_accl = AmxAccl(
        ## this where the debug output for the simulation will go
        dump_directory=args.dump_directory.as_posix()
    )

    # comment out if not out of order
    ## TODO: Check these numbers with actaul documentation
    ## TODO: Make sure that execution units, etc other CPU parameters are accurate to saphire rapids
    ## TODO: the Cache hierachy also needs to match saphire rapids CPUs
    
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


## to understand how the cache hierarchy is connected we can look at the python file for it 
## basically the same as a normal private l1, l2 hierarchy but is has a crossbar which connects 
# the amx simObject and the cpu simObject to the cache at the same points
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

# --- Setup the Workload ---
board.set_se_binary_workload(
    binary=BinaryResource(local_path=binary_path.as_posix())
)


# --- add some helpers to mark the simout ---
# ./[path to gem5] --debug-help gives more flag that we can use
def workbegin_handler():
    print(f"\n--- Start of AMX Region ---\n")
    m5.debug.flags["AMX"].enable()
    yield False  # Yielding true would end the simulaltion right away


def workend_handler():
    end_tick = m5.curTick()
    print(f"\n--- END of AMX Region ---\n")
    yield False 


# --- Setup and Run Simulator ---
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
