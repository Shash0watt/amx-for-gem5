# Implementation Plan

## Overivew

The goal is to get a gemm kernel that uses intel AMX intrinsics working in gem5.

The high level plan is to get a pesudo instruction to go through the O3 cpu pipeline, use the structures that the O3 cpu has to allow for proper out of order issue, then a AMX simObject can handle the execution semantics (keep track of 2d tile regsiters & acclerator configs, utilization etc) and use the LSQ to make sure that memory retires are handled properly.

##


some questions I have are:
- how will it detect what tiles are being used from the pesudo op & how will it know if our functional units are busy
- if we give the amx accelerator more control of issue order then how can we make sure that we are able to revert back to the correct state