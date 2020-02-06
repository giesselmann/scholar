# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Semantic Scholar S2 corpus meta data extraction
#
#  DESCRIPTION   : none
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019 Pay Giesselmann, Max Planck Institute for Molecular Genetics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Pay Giesselmann
# ---------------------------------------------------------------------------------
import sys, json, re
import signal
from tqdm import tqdm



def is_ascii(s):
    return all(ord(c) < 128 and ord(c) > 31 for c in s)




if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signal, frame: exit(0))
    required_keys = {'id', 'year', 'journalName', 'title', 'paperAbstract'}
    for line in sys.stdin:
        fields = json.loads(line)
        if (all(k in fields for k in required_keys) and
            all(fields[k] for k in required_keys)):
            j_name = re.sub(' +', ' ', fields['journalName'].replace('\n', ' ').replace('\t', ' '))
            title = re.sub(' +', ' ', fields['title'].replace('\n', ' ').replace('\t', ' '))
            abstract = re.sub(' +', ' ', fields['paperAbstract'].replace('\n', ' ').replace('\t', ' '))
            year = str(fields['year']) if str(fields['year']).isnumeric() else 'NA'
            if not all([is_ascii(s) for s in [j_name, title, abstract, year]]):
                continue
            print(json.dumps({'id': fields['id'], 'year' : year, 'journalName':j_name, 'title':title, 'paperAbstract':abstract}, sort_keys=True))
