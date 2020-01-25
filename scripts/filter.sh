#!/bin/bash
# $1 is repository with scripts
# $2 is batch file name
#pigz -d data/$2 -c | python3 ${1}/meta.py | pigz > data_filtered/$2
