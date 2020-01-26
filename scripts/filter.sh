#!/bin/bash
# $1 is repository with scripts
# $2 is batch file name
mkdir -p data_filtered
pigz -d data/$2 -c | python3 ${1}/filter.py | pigz > data_filtered/$2
