#!/bin/bash
cd ./shell
pwd

for fn in ./*; do
  sbatch $fn
  done