from m5.objects import (
    AmxAccl,
    AmxCoherentXBar,
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
    Rapids Golden Clove and connected to a Intel AMX Accelerator sharing its L1D.
    """

    def __init__(self, core_id: int, dump_directory: str):
        # These numbers are based of the Golden Clove microarch sheet and from the
        # intel optimisation manual's section on sapphire rappids
        super().__init__(
            core=X86O3CPU(
                cpu_id=core_id,
                fetchWidth=8,  # approximated by uop-cache delivery value
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

                # The FUPool limits how many loads and stores can issue from the IQ each cycle
                # cacheLoadPorts and cacheStorePorts limit how many packets are sent to the L1D each cycle
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
                # We are trying to model 3 loads and 2 stores per cycle, the default here
                # is 200 each (effectively no limit)
                cacheLoadPorts=3,
                cacheStorePorts=2,
            ),
            isa=ISA.X86,
        )

        self.core.amx_accl = AmxAccl(dump_directory=dump_directory)

        # Becase real load ports connect directly to the L1D, so the crossbarr shouldn't 
        # add any delay or serialize to 1 load per cycle. This is the basic crossbar but 
        # with no delay or serialising happening anymore
        self.l1d_amx_xbar = AmxCoherentXBar()

    # the cache hierarchy calls this method when we try and connect it to the CPU
    # here it's been modified to so that the xbar connects properly
    def connect_dcache(self, port):
        self.core.dcache_port = self.l1d_amx_xbar.cpu_side_ports
        self.core.amx_accl.mem_side = self.l1d_amx_xbar.cpu_side_ports
        self.l1d_amx_xbar.mem_side_ports = port
