from m5.objects import (
    AmxAccl,
    CoherentXBar,
    FP_ALU,
    FP_MultDiv,
    FUPool,
    IntALU,
    IQUnit,
    PredALU,
    RdWrPort,
    ReadPort,
    SIMD_Unit,
    System_Unit,
    WritePort,
    X86IntMultDiv,
    X86O3CPU,
)

from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.isas import ISA


class SapphireRapidsCore(BaseCPUCore):
    """An out-of-order X86 core with its parameters modified to match Sapphire
    Rapids Golden Clove and connected to a Intel AMX Accelerator sharing its L1D."""

    def __init__(self, core_id: int, dump_directory: str):
        # Golden Cove / Sapphire Rapids specs (Intel Optimization Manual ch. 2.4 and
        # the Sapphire Rapids core slide). Anything not set here is a gem5 default.
        super().__init__(
            core=X86O3CPU(
                cpu_id=core_id,
                fetchWidth=8,  # approximated by uop-cache delivery
                fetchBufferSize=32,
                fetchQueueSize=144,
                decodeWidth=6,
                renameWidth=6,
                dispatchWidth=6,  # approximated by allocation width value
                issueWidth=12,  # approximated by number of execution ports value
                commitWidth=8,
                numROBEntries=512,
                LQEntries=240,
                SQEntries=112,
                numPhysIntRegs=288,
                # Memory issue: separate AGUs, 3 for loads (ports 2, 3, 11) and 2 for
                # store addresses (ports 7, 8). This is gem5's DefaultX86FUPool with
                # only the three memory entries changed; its default is 4 units
                # shared by loads and stores (ReadPort/WritePort have count 0).
                instQueues=[
                    IQUnit(
                        fuPool=FUPool(
                            FUList=[
                                IntALU(),
                                X86IntMultDiv(),
                                FP_ALU(),
                                FP_MultDiv(),
                                ReadPort(count=3),
                                SIMD_Unit(),
                                PredALU(),
                                WritePort(count=2),
                                RdWrPort(count=0),
                                System_Unit(),
                            ]
                        )
                    )
                ],
                # L1D access: 3 loads and 2 stores per cycle, counted when a packet
                # is sent to the L1D (store writeback is after commit). gem5's
                # classic L1D has no port limit of its own, and the default here
                # is 200 each (effectively off).
                cacheLoadPorts=3,
                cacheStorePorts=2,
            ),
            isa=ISA.X86,
        )

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

    # the cache hierarchy calls this method when we try and connect it to the CPU
    # here it's been modified to so that the xbar connects properly
    def connect_dcache(self, port):
        self.core.dcache_port = self.l1d_amx_xbar.cpu_side_ports
        self.core.amx_accl.mem_side = self.l1d_amx_xbar.cpu_side_ports
        self.l1d_amx_xbar.mem_side_ports = port
