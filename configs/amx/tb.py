# System components
# Standard libraries
from pathlib import Path

import m5.debug

from m5.objects import AmxAccl, L2XBar

from gem5.components.cachehierarchies.classic.caches.l1dcache import (
    L1DCache,
)
from gem5.components.cachehierarchies.classic.caches.l1icache import (
    L1ICache,
)
from gem5.components.cachehierarchies.classic.caches.l2cache import (
    L2Cache,
)

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import DIMM_DDR5_4400

# Simulation components
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

"""
Usage: in root directory run
$ make
$ ./gem5.debug -rs amx/tb.py
"""

# Define the path to your compiled test binary containing the custom instructions
# binary_path = Path("configs/amx/binaries/load_test")
binary_path = Path("configs/amx/binaries/ooo_test")


# AMX-aware cache hierarchy
class AmxPrivateL1PrivateL2CacheHierarchy(
    PrivateL1PrivateL2CacheHierarchy
):
    def incorporate_cache(self, board):
        # Connect gem5's system/functional-access port.
        board.connect_system_port(self.membus.cpu_side_ports)

        # Connect the system memory bus to the memory controllers.
        for _, port in board.get_mem_ports():
            self.membus.mem_side_ports = port

        num_cores = board.get_processor().get_num_cores()

        # Existing crossbars between each core's L1 caches and private L2.
        self.l2buses = [
            L2XBar()
            for _ in range(num_cores)
        ]

        # New crossbars shared by each CPU data port and its AMX unit.
        self.amx_l1d_xbars = [
            L2XBar(width=64)
            for _ in range(num_cores)
        ]

        for i, cpu in enumerate(board.get_processor().get_cores()):
            # Create the private cache hierarchy for this core.
            l2_node = self.add_root_child(
                f"l2-cache-{i}",
                L2Cache(size=self._l2_size),
            )

            l1i_node = l2_node.add_child(
                f"l1i-cache-{i}",
                L1ICache(size=self._l1i_size),
            )

            l1d_node = l2_node.add_child(
                f"l1d-cache-{i}",
                L1DCache(size=self._l1d_size, assoc=12),
            )

            # Connect private L2 to the system memory bus.
            self.l2buses[i].mem_side_ports = l2_node.cache.cpu_side
            self.membus.cpu_side_ports = l2_node.cache.mem_side

            # Connect L1I and L1D toward the private L2.
            l1i_node.cache.mem_side = self.l2buses[i].cpu_side_ports
            l1d_node.cache.mem_side = self.l2buses[i].cpu_side_ports

            # The instruction path remains unchanged.
            cpu.connect_icache(l1i_node.cache.cpu_side)

            # Insert the new crossbar into the data path.
            amx_l1d_xbar = self.amx_l1d_xbars[i]

            cpu.connect_dcache(amx_l1d_xbar.cpu_side_ports)

            cpu_simobject = cpu.get_simobject()
            cpu_simobject.amx_accl.mem_side = (
                amx_l1d_xbar.cpu_side_ports
            )

            amx_l1d_xbar.mem_side_ports = l1d_node.cache.cpu_side

            # Keep the original page-table walker connections.
            self._connect_table_walker(i, cpu)

            # Keep the original interrupt connections.
            if board.get_processor().get_isa() == ISA.X86:
                int_req_port = self.membus.mem_side_ports
                int_resp_port = self.membus.cpu_side_ports
                cpu.connect_interrupt(int_req_port, int_resp_port)
            else:
                cpu.connect_interrupt()

        if board.has_coherent_io():
            self._setup_io_cache(board)


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
    core.core.amx_accl = AmxAccl()

    # comment out if not out of order
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
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# Setup Workload
board.set_se_binary_workload(
    binary=BinaryResource(local_path=binary_path.as_posix())
)

# ./[path to gem5] --debug-help gives more flag that we can use


def workbegin_handler():
    print("\n--- Start of AMX ROI ---\n")

    # Enable ExecAll here to trace instructions ONLY in your region of interest
    # This prevents the terminal from being flooded with standard C-library setup instructions.
    # m5.debug.flags["ExecAll"].enable()
    m5.debug.flags["Cache"].enable()
    m5.debug.flags["PseudoInst"].enable()
    m5.debug.flags["AMX"].enable()

    yield False  # Yielding False tells the simulator to continue running


def workend_handler():
    print("\n--- End of AMX ROI ---\n")

    # Disable tracing once the work is done
    # m5.debug.flags["ExecAll"].disable()
    m5.debug.flags["Cache"].disable()
    m5.debug.flags["PseudoInst"].disable()
    m5.debug.flags["AMX"].disable()

    yield False  # Yielding False tells the simulator to continue running


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
