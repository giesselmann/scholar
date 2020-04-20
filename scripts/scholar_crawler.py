# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Fetch citations from google scholar
#
#  DESCRIPTION   : none
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019-2020 Pay Giesselmann, Max Planck Institute for Molecular Genetics
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
import os, sys, argparse
import re, json
import timeit, time
import subprocess
from tqdm import tqdm
from tbselenium.tbdriver import TorBrowserDriver
from threading import Thread
from queue import Queue




# start tor as background process
class TorSession(object):
    def __init__(self, tbb_path, port=9050):
        tor_exec = args.tbb_path + '/Browser/TorBrowser/Tor/tor'
        tor_p = subprocess.Popen([tor_exec, 'SocksPort', str(port), 'DataDirectory', "~/.tor_{}".format(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tor_p_stdout = iter(tor_p.stdout)
        stdout = next(tor_p_stdout)
        while not b'100%' in stdout:
            try:
                stdout = next(tor_p_stdout)
            except StopIteration:
                print("Failed to start TOR on SOCKS {}".format(port), file=sys.stderr)
                tor_p.terminate()
                self.tor_p = None
                return
        print("Started tor on SOCKS {}".format(port), file=sys.stderr)
        self.tor_p = tor_p

    def __enter__(self):
        return self.tor_p

    def __exit__(self, type, value, traceback):
        if self.tor_p:
            self.tor_p.terminate()




#
class ScholarParser(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass




class ScholarRecord(object):
    def __init__(self):
        pass




if __name__ == '__main__':
    querry = 'Analysis of short tandem repeat expansions and their methylation state with nanopore sequencing'
    
