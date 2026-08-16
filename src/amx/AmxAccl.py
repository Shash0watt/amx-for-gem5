from m5.params import *
from m5.proxy import *

# from m5.SimObject import SimObject
from m5.objects.ClockedObject import ClockedObject


class AmxAccl(ClockedObject):
    type = "AmxAccl"
    cxx_header = "amx/amx_accl.hh"
    cxx_class = "gem5::AmxAccl"

    mem_side = RequestPort("AMX memory-side port for tile loads and stores")

    dump_directory = Param.String(
        "amx_debug", "directory for AMX state dump files"
    )

    # intel amx optimization manual table 20-2 reports about 204 cycles.
    config_latency = Param.Cycles(
        204, "minimum issue-to-completion latency for tile configuration"
    )

    load_issue_throughput = Param.Cycles(
        8, "minimum cycles between issued tile load instructions"
    )

    load_latency = Param.Cycles(
        45, "minimum issue-to-completion latency for tile load instructions"
    )

    dp_issue_throughput = Param.Cycles(
        16, "minimum cycles between issued tile dot product instructions"
    )

    dp_latency = Param.Cycles(
        52, "issue-to-completion latency for TDPBF16PS instructions"
    )

    # TILEZERO has no issue restriction in Intel's timing table. A zero-cycle
    # initiation interval lets independent zero instructions issue together.
    zero_issue_throughput = Param.Cycles(
        0, "minimum cycles between issued tile zero instructions"
    )

    zero_latency = Param.Cycles(
        16, "issue-to-completion latency for tile zero instructions"
    )

    store_issue_throughput = Param.Cycles(
        16, "minimum cycles between issued tile store instructions"
    )

    # Store completion will ultimately depend on the memory system. Keeping a
    # configurable minimum supports the common max(minimum, memory completion)
    # path without inventing a fixed hardware latency.
    store_latency = Param.Cycles(
        0, "minimum issue-to-completion latency for tile store instructions"
    )
