GEM5_PATH = .
ISA = x86

# compilers
CC = gcc
CXX = g++
PYTHON = python3

# compiler and linker flags
INCLUDES = -I$(GEM5_PATH)/include -I$(GEM5_PATH)/configs/amx/tests
LDFLAGS = -L$(GEM5_PATH)/util/m5/build/$(ISA)/out -lm5

# source files and targets
SRCS_CPP := $(shell find configs/amx/tests -name '*.cpp' ! -name 'tile_load_benchmark.cpp')
SRCS_C   := $(shell find configs/amx/tests -name '*.c')

TARGETS_CPP := $(patsubst configs/amx/tests/%.cpp,configs/amx/binaries/%,$(SRCS_CPP))
TARGETS_C   := $(patsubst configs/amx/tests/%.c,configs/amx/binaries/%,$(SRCS_C))
TARGETS     := $(TARGETS_CPP) $(TARGETS_C)

ACCURACY_RUNNER = configs/amx/tests/run_accuracy_tests.py

.PHONY: all test update-correct clean

all: $(TARGETS)

# pattern rule to compile c++ files
configs/amx/binaries/%: configs/amx/tests/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -o $@ $< $(INCLUDES) $(LDFLAGS)

# pattern rule to compile c files
configs/amx/binaries/%: configs/amx/tests/%.c
	@mkdir -p $(dir $@)
	$(CC) -o $@ $< $(INCLUDES) $(LDFLAGS)

test:
	$(PYTHON) $(ACCURACY_RUNNER)

update-correct:
	$(PYTHON) $(ACCURACY_RUNNER) --update-correct

clean:
	rm -rf configs/amx/binaries

