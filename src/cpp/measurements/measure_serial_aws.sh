#!/bin/bash
rm -f measure_serial.txt
for j in `seq 5 1 15` `seq 16 2 30` `seq 35 5 60` `seq 70 10 120`;
do
    ./sicc measure_serial_aws.txt $j 8 32 32 32 1 15
    ./sicc measure_serial_aws.txt $j 8 32 32 32 1 15
done
