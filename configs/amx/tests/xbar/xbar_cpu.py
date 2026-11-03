from m5.objects import AmxAccl, AmxXbar, X86O3CPU

from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.isas import ISA


class XbarCPU(BaseCPUCore):
    def __init__(self, core_id: int, dump_directory: str):
        super().__init__(core=X86O3CPU(cpu_id=core_id), isa=ISA.X86)

        self.core.amx_accl = AmxAccl(dump_directory=dump_directory)

        # Same mux as SapphireRapidsCore.
        self.l1d_amx_xbar = AmxXbar()

    def connect_dcache(self, port):
        self.core.dcache_port = self.l1d_amx_xbar.cpu_side
        self.core.amx_accl.mem_side = self.l1d_amx_xbar.accl_side
        self.l1d_amx_xbar.mem_side = port
