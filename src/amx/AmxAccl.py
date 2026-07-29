from m5.params import *
from m5.proxy import *
# from m5.SimObject import SimObject
from m5.objects.ClockedObject import ClockedObject


class AmxAccl(ClockedObject):
    type = "AmxAccl"
    cxx_header = "amx/amx_accl.hh"
    cxx_class = "gem5::AmxAccl"

    mem_side = RequestPort(
        "AMX memory-side port for tile loads and stores"
    )

    # intel amx optimization manual table 20-2 reports about 204 cycles.
    config_latency = Param.Cycles(
        204, "minimum issue-to-completion latency for tile configuration"
    )
