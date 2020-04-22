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
from tbselenium.tbdriver import TorBrowserDriver




# start tor as background process
class TorSession(object):
    def __init__(self, tbb_path, port=9050):
        tor_exec = tbb_path + '/Browser/TorBrowser/Tor/tor'
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




# parse google scholar querries
class ScholarParser(object):
    def __init__(self, driver):
        self._driver = driver

    def record_from_querry(self, querry):
        querry_url = 'https://scholar.google.com/scholar?hl=de&as_sdt=0%2C5&q={}'.format(
            '+'.join(querry.split()))
        self._driver.get(querry_url)
        title = self._driver.find_element_by_class_name('gs_rt').text
        return ScholarRecord(title, 0)




class ScholarRecord(object):
    def __init__(self, title='', citations=0):
        self._title = title
        self._citations = citations

    @property
    def title(self):
        return self._title

    @property
    def citations(self):
        return self._citations




if __name__ == '__main__':
    tbb_path = '/home/pay/tor/tor-browser_en-US/'
    socks_port = 9050
    querry = 'Analysis of short tandem repeat expansions and their methylation state with nanopore sequencing'
    with TorSession(tbb_path, socks_port) as ts, \
         TorBrowserDriver(tbb_path, socks_port=socks_port) as driver:
          parser = ScholarParser(driver)
          record = parser.record_from_querry(querry)
    print(record.title)
    print(record.citations)
