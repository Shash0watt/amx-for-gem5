In x86 there is a unused opcode
- following the 2 byte opcode escape prefix 0x0F
- the opcode 0x04 is left unused
by default gem5 uses the lower 8 bits in the immediate feild. 
so we have enough space.

 m5ops can a also be triggered using the magic instruciton address space kinda like MMIO