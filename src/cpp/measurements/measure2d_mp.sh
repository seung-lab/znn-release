#!/bin/bash
for j in `seq 5 5 30` `seq 40 10 60` `seq 80 20 120`;
do
    ./measure2dmp $1 $j 8 48 48 1 $2 $3
done
