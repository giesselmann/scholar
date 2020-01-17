# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Fetch title and abstract from bioRxiv
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
import timeit
import subprocess
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from tbselenium.tbdriver import TorBrowserDriver
from pyvirtualdisplay import Display




def get_page(driver, page=0):
    page_url = 'https://www.biorxiv.org/content/early/recent?page={}'.format(page)
    driver.get(page_url)
    lists = driver.find_elements_by_class_name("highwire-article-citation-list")
    records = {}
    print("loading titles")
    t0 = timeit.default_timer()
    for l in lists:
        title = l.find_element_by_class_name('highwire-list-title').text
        year = re.search('[0-9]{4}', title).group(0)
        articles = l.find_elements_by_class_name("clearfix")
        for article in articles:
            title = article.find_element_by_class_name("highwire-cite-title").text
            authors = [x.strip() for x in article.find_element_by_class_name("highwire-cite-authors").text.split(',')]
            article_lnk = article.find_element_by_class_name('highwire-cite-linked-title').get_attribute('href')
            records[article_lnk] = {'title': title, 'authors': authors, 'year' : year}
    t1 = timeit.default_timer()
    for key, value in list(records.items()):
        driver.get(key)
        abstract = driver.find_element_by_class_name('abstract').text
        abstract = re.sub(' +', ' ', ' '.join(abstract.split('\n')[1:]))
        value['abstract'] = abstract
        records[key] = json.dumps(value, sort_keys=True)
    t2 = timeit.default_timer()
    print("load titles: {:.2f}".format(t1-t0))
    print("load abstracts {:.2f}".format(t2-t1))
    #print(records)
    #print([l.find_element_by_class_name('highwire-list-title').text for l in lists])
    #links = driver.find_elements_by_class_name("highwire-cite-linked-title")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BioRxiv meta data crawler')
    parser.add_argument('tbb_path', help='Path to Tor browser')
    args = parser.parse_args()
    # virtual display
    display = Display(visible=0, size=(1024, 2048))
    display.start()
    # start tor session
    tor_exec = args.tbb_path + '/Browser/TorBrowser/Tor/tor'
    tor_p = subprocess.Popen([tor_exec, 'SocksPort', '9050'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tor_p_stdout = iter(tor_p.stdout)
    stdout = next(tor_p_stdout)
    while not b'100%' in stdout:
        stdout = next(tor_p_stdout)
    print("Tor started successfully")
    with TorBrowserDriver(args.tbb_path, socks_port=9050) as driver:
        #driver.implicitly_wait(1) # seconds
        get_page(driver, page=0)
    tor_p.terminate()
