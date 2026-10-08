#!/usr/bin/env bash
# Combine the 4 track.py modes for one video into a labelled 2x2 GIF.
#   for m in boxes path both compare; do python track.py --video swing.mp4 --mode $m; done
#   scripts/make_grid.sh out/swing   assets/modes_grid.gif
# Needs ffmpeg (brew install ffmpeg).
set -euo pipefail
STEM=${1:?usage: make_grid.sh out/<video_stem> [output.gif] [width_per_tile]}
OUT=${2:-assets/modes_grid.gif}
W=${3:-220}
FPS=${FPS:-10}
FONT=${FONT:-}
if [ -z "$FONT" ]; then
  for f in /System/Library/Fonts/Supplemental/Arial\ Bold.ttf /Library/Fonts/Arial\ Bold.ttf \
           /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
           /usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf; do
    [ -f "$f" ] && FONT="$f" && break
  done
fi
lab() { echo "scale=${W}:-2,drawtext=fontfile='${FONT}':text='$1':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6"; }
ffmpeg -v error -y \
  -i "${STEM}_boxes.mp4" -i "${STEM}_path.mp4" -i "${STEM}_both.mp4" -i "${STEM}_compare.mp4" \
  -filter_complex "\
[0:v]$(lab 'boxes')[a];[1:v]$(lab 'path')[b];[2:v]$(lab 'both')[c];[3:v]$(lab 'compare')[d];\
[a][b]hstack[top];[c][d]hstack[bot];[top][bot]vstack,fps=${FPS},split[x][y];\
[x]palettegen=max_colors=128[p];[y][p]paletteuse=dither=bayer" \
  -loop 0 "$OUT"
echo "saved $OUT ($(du -h "$OUT" | cut -f1))"
