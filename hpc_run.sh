#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=fc_hydrology
#
# Partition:
#SBATCH --partition=savio2
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
## Command(s) to run:
source activate py2k_model
python hpc_newman.py
DATE=`date +%Y-%m-%d:%H:%M:%S`
TITLESTR="NEWMAN_$DATE"
x='Analysis is finished.'
sendmail daviddralle@gmail.com << EOF
subject:$TITLESTR
$x
EOF
