from m5.objects import L2XBar

from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.cachehierarchies.abstract_cache_hierarchy import (
    AbstractCacheHierarchy,
)
from gem5.components.cachehierarchies.classic.caches.l1dcache import (
    L1DCache,
)
from gem5.components.cachehierarchies.classic.caches.l1icache import (
    L1ICache,
)
from gem5.components.cachehierarchies.classic.caches.l2cache import (
    L2Cache,
)
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.isas import ISA
from gem5.utils.override import overrides


class AmxPrivateL1PrivateL2CacheHierarchy(
    PrivateL1PrivateL2CacheHierarchy
):
    """Connect each CPU and its AMX accelerator to a shared private L1D."""

    @overrides(AbstractCacheHierarchy)
    def incorporate_cache(self, board: AbstractBoard) -> None:
        # Set up the system port for functional access from the simulator.
        board.connect_system_port(self.membus.cpu_side_ports)

        for _, port in board.get_mem_ports():
            self.membus.mem_side_ports = port

        self.l2buses = [
            L2XBar() for i in range(board.get_processor().get_num_cores())
        ]

        # AMX change: create one CPU/AMX-to-L1D crossbar per core.
        self.amx_l1d_xbars = [
            L2XBar(width=64)
            for i in range(board.get_processor().get_num_cores())
        ]

        for i, cpu in enumerate(board.get_processor().get_cores()):
            l2_node = self.add_root_child(
                f"l2-cache-{i}", L2Cache(size=self._l2_size)
            )
            l1i_node = l2_node.add_child(
                f"l1i-cache-{i}", L1ICache(size=self._l1i_size)
            )
            # for AMX.. use 12-way associativity for the 48 KiB L1D.
            l1d_node = l2_node.add_child(
                f"l1d-cache-{i}", L1DCache(size=self._l1d_size, assoc=12)
            )

            self.l2buses[i].mem_side_ports = l2_node.cache.cpu_side
            self.membus.cpu_side_ports = l2_node.cache.mem_side

            l1i_node.cache.mem_side = self.l2buses[i].cpu_side_ports
            l1d_node.cache.mem_side = self.l2buses[i].cpu_side_ports

            cpu.connect_icache(l1i_node.cache.cpu_side)

            # for AMX.. connect CPU and AMX to L1D through a crossbar.
            amx_l1d_xbar = self.amx_l1d_xbars[i]
            # connect the cpu to the xbar
            cpu.connect_dcache(amx_l1d_xbar.cpu_side_ports)

            # get the amx simobject and then connect it to the xbar
            cpu.get_simobject().amx_accl.mem_side = (
                amx_l1d_xbar.cpu_side_ports
            )

            # connect teh xbar to the l1 cache
            amx_l1d_xbar.mem_side_ports = l1d_node.cache.cpu_side

            self._connect_table_walker(i, cpu)

            if board.get_processor().get_isa() == ISA.X86:
                int_req_port = self.membus.mem_side_ports
                int_resp_port = self.membus.cpu_side_ports
                cpu.connect_interrupt(int_req_port, int_resp_port)
            else:
                cpu.connect_interrupt()

        if board.has_coherent_io():
            self._setup_io_cache(board)
