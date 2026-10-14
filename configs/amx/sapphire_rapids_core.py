from m5.objects import (
    AmxAccl,
    CoherentXBar,
)

from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_core import SimpleCore
from gem5.isas import ISA


class SapphireRapidsCore(SimpleCore):
    """An out-of-order X86 core with it's parameter's modified to match saphire rapids and connected to a Intel AMX Accelerator sharing its L1D."""

    def __init__(self, core_id: int, dump_directory: str):
        super().__init__(cpu_type=CPUTypes.O3, core_id=core_id, isa=ISA.X86)

        self.core.amx_accl = AmxAccl(dump_directory=dump_directory)

        # Coherent - becauuse the 03 cpu wants snooping.
        # Zero latency - real load ports connect directly to the L1D, so the xbar should
        # only add contention.
        # Bandwidth (not precisely modeled) - Sapphire Rapids does 3 loads/cycle (2 for
        # AVX-512/AMX). Packets with data occupy a path for at least 1 cycle, and each
        # requester has one response path, so we get 1 load/cycle for the CPU and 1 line
        # (64 B)/cycle for AMX. Writes share one request path: 1 write/cycle combined.
        self.l1d_amx_xbar = CoherentXBar(
            width=64,
            header_latency=0,
            frontend_latency=0,
            forward_latency=0,
            response_latency=0,
            snoop_response_latency=0,
        )

        # Golden Cove / Sapphire Rapids specs (Intel Optimization Manual ch. 2.4 and
        # the Sapphire Rapids core slide). Anything not set here is a gem5 default.
        self.core.fetchWidth = 8  # approximated by uop-cache delivery
        self.core.fetchBufferSize = 32
        self.core.fetchQueueSize = 144

        self.core.decodeWidth = 6
        self.core.renameWidth = 6
        self.core.dispatchWidth = 6  # approximated by allocation width value
        self.core.issueWidth = 12  # approximated by number of execution ports value
        self.core.commitWidth = 8

        self.core.numROBEntries = 512
        self.core.LQEntries = 240
        self.core.SQEntries = 112

        self.core.numPhysIntRegs = 288

    # the cache hierarchy calls this method when we try and connect it to the CPU
    # here it's been modified to so that the xbar connects properly
    def connect_dcache(self, port):
        self.core.dcache_port = self.l1d_amx_xbar.cpu_side_ports
        self.core.amx_accl.mem_side = self.l1d_amx_xbar.cpu_side_ports
        self.l1d_amx_xbar.mem_side_ports = port
