# Build the difference map animations and figures.
#
#   make            # frames/frames.mp4 from the pngs already in frames/
#   make frames     # (re)generate the pngs with sudoku_dm.py
#   make gif        # a looping gif of the same frames
#   make sensitivity bench
#   make clean

PYTHON  ?= python3
FFMPEG  ?= ffmpeg
CONVERT ?= convert

FRAME_DIR ?= frames
PREFIX    ?= it-
FPS       ?= 30
# integer upscale of the 256x256 frames; nearest neighbour, so the digits stay crisp
SCALE     ?= 2
CRF       ?= 18

VIDEO := $(FRAME_DIR)/frames.mp4
GIF   := $(FRAME_DIR)/frames.gif
PNGS  := $(wildcard $(FRAME_DIR)/$(PREFIX)*.png)

# arguments for the run that writes the frames
PUZZLE     ?= 006050002000000090000008300095026000002030001400700000620000005001000030530060100
ITERATIONS ?= 4000
SOLVER_ARGS ?= -i $(ITERATIONS) -d $(FRAME_DIR) --prefix $(PREFIX) $(PUZZLE)

BENCH_IN  ?= test-xtrm.csv
BENCH_OUT ?= bench-$(BENCH_IN)
SENSITIVITY := images/sensitivity_dm.png

.PHONY: all video gif frames sensitivity bench clean clean-frames clean-video

all: video

video: $(VIDEO)

# -framerate before -i sets the input rate, i.e. one png per 1/FPS second
$(VIDEO): $(PNGS)
	@test -n "$(PNGS)" || { echo "no $(FRAME_DIR)/$(PREFIX)*.png; run 'make frames' first"; exit 1; }
	$(FFMPEG) -y -framerate $(FPS) -pattern_type glob -i '$(FRAME_DIR)/$(PREFIX)*.png' \
		-vf "scale=iw*$(SCALE):ih*$(SCALE):flags=neighbor" \
		-c:v libx264 -preset slow -crf $(CRF) -pix_fmt yuv420p -movflags +faststart $@

gif: $(GIF)

$(GIF): $(PNGS)
	@test -n "$(PNGS)" || { echo "no $(FRAME_DIR)/$(PREFIX)*.png; run 'make frames' first"; exit 1; }
	$(CONVERT) -delay $$((100 / $(FPS))) -loop 0 $(PNGS) -layers optimize $@

# solve a puzzle and write one png per iteration.  phony: the frame count
# depends on the run, so let the user ask for it explicitly.
frames:
	$(PYTHON) sudoku_dm.py $(SOLVER_ARGS)

sensitivity: $(SENSITIVITY)

$(SENSITIVITY): sensitivity_dm.py $(BENCH_IN)
	$(PYTHON) sensitivity_dm.py $(BENCH_IN) -o $@

bench: benchmark_dm.py $(BENCH_IN)
	$(PYTHON) benchmark_dm.py $(BENCH_IN) -o $(BENCH_OUT)

clean-video:
	rm -f $(VIDEO) $(GIF)

clean-frames:
	rm -f $(FRAME_DIR)/$(PREFIX)*.png

clean: clean-video
	rm -rf __pycache__
