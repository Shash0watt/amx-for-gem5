GEM5_PATH = .
ISA = x86

# compilers
CC = gcc
CXX = g++

# compiler and linker flags
INCLUDES = -I$(GEM5_PATH)/include
LDFLAGS = -L$(GEM5_PATH)/util/m5/build/$(ISA)/out -lm5

# source files and targets
SRCS_CPP = $(wildcard configs/amx/tests/*.cpp)

TARGETS = $(patsubst configs/amx/tests/%.cpp,configs/amx/binaries/%,$(SRCS_CPP))

all: $(TARGETS)

# pattern rule to compile c++ files
configs/amx/binaries/%: configs/amx/tests/%.cpp
	@mkdir -p configs/amx/binaries
	$(CXX) -o $@ $< $(INCLUDES) $(LDFLAGS)

# pattern rule to compile c files
configs/amx/binaries/%: configs/amx/tests/%.c
	@mkdir -p configs/amx/binaries
	$(CC) -o $@ $< $(INCLUDES) $(LDFLAGS)

clean:
	rm -rf configs/amx/binaries