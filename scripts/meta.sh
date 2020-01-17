#!/bin/bash
/project/bioinf_meissner/bin/pigz -d data/$1 -c | python3 meta.py | /project/bioinf_meissner/bin/pigz > meta/$1
/project/bioinf_meissner/bin/pigz -d data/$1 -c | python3 abstract.py | /project/bioinf_meissner/bin/pigz > abstract/$1
