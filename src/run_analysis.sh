#!/bin/sh
# Runs 18 chains for 20000 samples

seq 1 18 | xargs -n 1 -P 20 -I[] ./timeVaryingSEIR sample num_samples=20000 num_warmup=1000 data file=data.json random seed=[] output file=../results/outputs/output_[].csv

stansummary -s5 ../results/outputs/*.csv > ../results/stansummary.txt
