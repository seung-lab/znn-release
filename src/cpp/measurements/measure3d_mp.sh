#!/bin/bash
for j in `seq 5 5 30` `seq 40 10 60` `seq 80 20 120`;
do
    ./measure3dmp $1 $j 6 12 12 12 $2 $3
done
