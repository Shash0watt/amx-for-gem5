# AMX intrinsic results: hardware vs. gem5


## TDPBF16PS (dot_product_test.cpp)

| Test             | Hardware | gem5  | Match |
| ---------------- | -------- | ----- | ----- |
| chain through C  | 16.09    | 52.19 | no    |
| chain through A  | 52.05    | 52.19 | yes   |
| chain through B  | 52.05    | 52.19 | yes   |
| six accumulators | 16.09    | 16.21 | yes   |
