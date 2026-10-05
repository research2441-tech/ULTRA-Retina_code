#!/bin/sh
# Create the working folders used by the scripts and move the shipped result files into runs/.
mkdir -p runs logs figs
for f in pred_*.npz *.json tables.md policy_val_s0_partial.log; do [ -f "$f" ] && mv "$f" runs/; done
echo "folders ready"
