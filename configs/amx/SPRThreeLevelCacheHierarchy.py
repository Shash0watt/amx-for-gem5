"""
This module contains a Sapphire Rapids-like three-level cache hierarchy with
private L1 caches, private L2 caches, and a shared non-inclusive L3 cache.
"""

from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.cachehierarchies.classic.abstract_classic_cache_hierarchy import (
    AbstractClassicCacheHierarchy,
)

from gem5.components.cachehierarchies.classic.caches.l1dcache import L1DCache
from gem5.components.cachehierarchies.classic.caches.l1icache import L1ICache
from gem5.components.cachehierarchies.classic.caches.l2cache import L2Cache
from gem5.components.cachehierarchies.classic.caches.mmu_cache import MMUCache

from gem5.isas import ISA

from m5.objects import (
    BadAddr,
    Cache,
    L2XBar,
    SystemXBar,
    SubSystem,
)


class SPR_Private_L1_Private_L2_Shared_L3_Cache_Hierarchy(
    AbstractClassicCacheHierarchy
):

    # Defaults are Sapphire Rapids (Xeon Platinum 8468) sizes and ways. gem5
    # needs a power-of-two set count, so the 105 MiB 15-way L3 is rounded up to
    # the nearest size that has one (120 MiB).
    def __init__(
        self,
        l1d_size="48KiB",
        l1i_size="32KiB",
        l2_size="2MiB",
        l3_size="120MiB",
        l1d_assoc=12,
        l1i_assoc=8,
        l2_assoc=16,
        l3_assoc=15,
    ):
        AbstractClassicCacheHierarchy.__init__(self)

        # Save the sizes to use later. We have to use leading underscores
        # because the SimObject (SubSystem) does not have these attributes as
        # parameters.
        self._l1d_size = l1d_size
        self._l1i_size = l1i_size
        self._l2_size = l2_size
        self._l3_size = l3_size
        self._l1d_assoc = l1d_assoc
        self._l1i_assoc = l1i_assoc
        self._l2_assoc = l2_assoc
        self._l3_assoc = l3_assoc

        # Use a high-bandwidth system crossbar.
        self.membus = SystemXBar(width=64)
        # For FS mode
        self.membus.badaddr_responder = BadAddr()
        self.membus.default = self.membus.badaddr_responder.pio

        # We can't create the caches yet, because we don't know how many cores there are.

    # To connect the memory system to the caches
    def get_mem_side_port(self):
        return self.membus.mem_side_ports

    # For FS mode. This is a coherent port.
    def get_cpu_side_port(self):
        return self.membus.cpu_side_ports

    # This is where the bulk of the work happens.
    # The board calls this function after it has created the processor and
    # memory system. The cache hierarchy is responsible for connecting things.
    def incorporate_cache(self, board):
        # Connect the system port to the memory system.
        board.connect_system_port(self.membus.cpu_side_ports)

        # Connect the memory system to the memory port on the board.
        for _, port in board.get_memory().get_mem_ports():
            self.membus.mem_side_ports = port

        # Create an L3 crossbar (64 bytes wide, one line per cycle)
        self.l3_bus = L2XBar(width=64)

        self.clusters = [
            self._create_core_cluster(
                core, self.l3_bus, board.get_processor().get_isa()
            )
            for core in board.get_processor().get_cores()
        ]

        self.l3_cache = L3Cache(size=self._l3_size, assoc=self._l3_assoc)

        # Connect the L3 cache to the system crossbar and L3 crossbar
        self.l3_cache.mem_side = self.membus.cpu_side_ports
        self.l3_cache.cpu_side = self.l3_bus.mem_side_ports

        if board.has_coherent_io():
            self._setup_io_cache(board)

    def _create_core_cluster(self, core, l3_bus, isa):
        """
        Create a core cluster with the given core.
        """
        cluster = SubSystem()
        cluster.l1dcache = L1DCache(size=self._l1d_size, assoc=self._l1d_assoc)
        cluster.l1icache = L1ICache(
            size=self._l1i_size, assoc=self._l1i_assoc, writeback_clean=False
        )
        # Clean L2 evictions must be written back, otherwise the mostly
        # exclusive L3 would only ever receive dirty lines.
        cluster.l2cache = L2Cache(
            size=self._l2_size, assoc=self._l2_assoc, writeback_clean=True
        )

        cluster.iptw_cache = MMUCache(size="8KiB", writeback_clean=False)
        cluster.dptw_cache = MMUCache(size="8KiB", writeback_clean=False)

        cluster.l2_bus = L2XBar(width=64)

        # Connect the core to the caches
        core.connect_icache(cluster.l1icache.cpu_side)
        core.connect_dcache(cluster.l1dcache.cpu_side)
        core.connect_walker_ports(
            cluster.iptw_cache.cpu_side, cluster.dptw_cache.cpu_side
        )

        # Connect the caches to the L2 bus
        cluster.l1dcache.mem_side = cluster.l2_bus.cpu_side_ports
        cluster.l1icache.mem_side = cluster.l2_bus.cpu_side_ports
        cluster.iptw_cache.mem_side = cluster.l2_bus.cpu_side_ports
        cluster.dptw_cache.mem_side = cluster.l2_bus.cpu_side_ports

        cluster.l2cache.cpu_side = cluster.l2_bus.mem_side_ports

        cluster.l2cache.mem_side = l3_bus.cpu_side_ports

        if isa == ISA.X86:
            int_req_port = self.membus.mem_side_ports
            int_resp_port = self.membus.cpu_side_ports
            core.connect_interrupt(int_req_port, int_resp_port)
        else:
            core.connect_interrupt()

        return cluster

    # from exmaple (unused in SE mode on X86Board
    def _setup_io_cache(self, board: AbstractBoard) -> None:
        """Create a cache for coherent I/O connections"""
        self.iocache = Cache(
            assoc=8,
            tag_latency=50,
            data_latency=50,
            response_latency=50,
            mshrs=20,
            size="1KiB",
            tgts_per_mshr=12,
            addr_ranges=board.mem_ranges,
        )
        self.iocache.mem_side = self.membus.cpu_side_ports
        self.iocache.cpu_side = board.get_mem_side_coherent_io_port()


class L3Cache(Cache):
    def __init__(self, size, assoc):
        super().__init__()
        self.size = size
        self.assoc = assoc
        self.tag_latency = 20
        self.data_latency = 20
        self.response_latency = 1
        self.mshrs = 20
        self.tgts_per_mshr = 12
        self.writeback_clean = False
        # Intel server L3s have been non-inclusive since Skylake-SP: L2 misses
        # fill straight into L2, and L3 is filled mostly by L2 evictions (a
        # victim cache). SPR keeps this design, since a 2 MiB L2 is bigger
        # than a core's 1.875 MiB L3 slice and an inclusive L3 would waste
        # most of it on duplicates. mostly_excl is gem5's closest match: L3
        # doesn't allocate on fills headed to L2, and drops clean lines it
        # hands back up (SPR sometimes keeps them; gem5 always drops them).
        self.clusivity = "mostly_excl"