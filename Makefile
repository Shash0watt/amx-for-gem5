GEM5_PATH = .
ISA = x86

# Compilers
CC = gcc
CXX = g++

# Compiler and Linker Flags
INCLUDES = -I$(GEM5_PATH)/include
LDFLAGS = -L$(GEM5_PATH)/util/m5/build/$(ISA)/out -lm5

TARGET = configs/amx/load_test

all: $(TARGET)

# Pattern rule to compile C++ files
%: %.cpp
	$(CXX) -o $@ $< $(INCLUDES) $(LDFLAGS)

# Pattern rule to compile C files
%: %.c
	$(CC) -o $@ $< $(INCLUDES) $(LDFLAGS)

clean:
	rm -f $(TARGET)