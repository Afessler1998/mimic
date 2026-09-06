CXX      = g++
CXXFLAGS = -std=c++23 -Wall -Wextra
DEPFLAGS = -MMD -MP

# -O2 on toml++'s template instantiations dominates build time, so iterate at
# -O0 and opt in to optimisation with `make release`.
OPT ?= -O0 -g
CXXFLAGS += $(OPT)

# the library never throws, enforced rather than remembered. the tools link
# OpenCV, which does, so they cannot make the same promise.
LIB_CXXFLAGS  = $(CXXFLAGS) -fno-exceptions
TOOL_CXXFLAGS = $(CXXFLAGS)

LIB_DIR  = server_lib
TOOL_DIR = tool_lib
OBJ_DIR = obj
BIN_DIR = bin
OUT_LIB = lib

LIB      = $(OUT_LIB)/libmocapserver.so
LIB_SRCS = $(wildcard $(LIB_DIR)/*.cpp)
LIB_OBJS = $(LIB_SRCS:$(LIB_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# each tool is a standalone binary linking the shared library
TOOLS = lens_calibration stereo_calibration
BINS  = $(TOOLS:%=$(BIN_DIR)/%)

INCLUDES = -I$(LIB_DIR)

# tool_lib is only what more than one tool needs: the device to host bridge and
# the intrinsics that stereo calibration reads back. anything belonging to a
# single tool lives in that tool's own directory.
#
# the library itself stays free of OpenCV and cudart: it hands out device
# pointers and lets each tool decide whether it wants host memory.
SHARED_TOOL_SRCS = $(wildcard $(TOOL_DIR)/*.cpp)
TOOL_INCLUDES = -I$(TOOL_DIR) -isystem /usr/include/opencv5 -isystem /opt/cuda/include
# OpenCV 5 renamed calib3d to calib, and moved the chessboard functions into
# objdetect. the 3d module is not built in this package at all.
TOOL_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui \
            -lopencv_calib -lopencv_objdetect \
            -L/opt/cuda/lib64 -lcudart
LIBS     = -lavcodec -lavutil
# library symbols are hidden unless marked MOCAP_API, so the exported surface
# is what session.hpp declares rather than every template instantiation
VISIBILITY = -fvisibility=hidden -fvisibility-inlines-hidden
# $$ORIGIN lets the binaries find the .so relative to themselves, uninstalled
RPATH    = -Wl,-rpath,'$$ORIGIN/../$(OUT_LIB)'

DEPS = $(LIB_OBJS:.o=.d) $(TOOLS:%=$(OBJ_DIR)/%.o.d)

.PHONY: all release clean run_lens run_stereo

all: $(LIB) $(BINS)

$(LIB): $(LIB_OBJS) | $(OUT_LIB)
	$(CXX) $(LIB_CXXFLAGS) -shared $^ -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(LIB_CXXFLAGS) $(DEPFLAGS) -fPIC $(VISIBILITY) $(INCLUDES) -c $< -o $@

.SECONDEXPANSION:
$(BIN_DIR)/%: %/main.cpp $$(wildcard %/*.cpp) $(SHARED_TOOL_SRCS) $(LIB) | $(BIN_DIR)
	$(CXX) $(TOOL_CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $(TOOL_INCLUDES) -I$* \
	  $< $(filter-out $*/main.cpp,$(wildcard $*/*.cpp)) $(SHARED_TOOL_SRCS) -o $@ \
	  -L$(OUT_LIB) -lmocapserver $(TOOL_LIBS) $(RPATH)

$(OBJ_DIR) $(BIN_DIR) $(OUT_LIB):
	@mkdir -p $@

run_lens: $(BIN_DIR)/lens_calibration
	@./$<

run_stereo: $(BIN_DIR)/stereo_calibration
	@./$<

release:
	@$(MAKE) --no-print-directory OPT="-O2 -g" clean all

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(OUT_LIB)

-include $(DEPS)
