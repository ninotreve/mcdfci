#!/bin/bash
for bond in $(seq 1.5 0.05 2.5) $(seq 2.6 0.1 4.5)
do
    a=$(echo "$bond * 0.52917721067" | bc)
    cp input.dat input_new.dat
    sed -i "s/1.12079733/${a}/g" input_new.dat
    psi4 input_new.dat output.dat
    cdfci_omp n2_ccpvqz.json > n2_${bond}.out
done
date
