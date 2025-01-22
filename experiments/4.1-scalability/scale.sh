#!/bin/bash
path="experiments/4.1-scalability"
for system in n2_ccpvdz n2_ccpvtz n2_ccpvqz
do
    for N in 256 128 64 32 16 8 4 2 1
    do
        python3 test_python/gen_thread.py ${system} ${N} ${path}
        cdfci_omp "${path}/${system}_${N}.json"
    done
done
