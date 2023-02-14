#! /usr/bin/env python3
""" Script to download public logs """

import os
import glob
import json
import sys
import time

import requests

def browseData():
    stepSize = 100
    URLPart1 = 'https://review.px4.io/browse_data_retrieval?draw=1 \
            &order%5B0%5D%5Bcolumn%5D=1&order%5B0%5D%5Bdir%5D=desc'
    
    URL = URLPart1 + \
            '&start=0&length=1&search%5Bvalue%5D=&search%5Bregex%5D=false'

    pageSize = requests.get(url=URL).json()['recordsTotal']

    db_entries_list = []
    for i in range(0, pageSize, stepSize):
        URL = URLPart1 + \
            "&start="+str(i)+"&length="+str(stepSize)+"&search%5Bvalue%5D=&search%5Bregex%5D=false"
        dataInfo = requests.get(url=URL).json()
        for elements in dataInfo['data']:
            logInfo = elements[1].split('plot_app?log=')[1].split('">')[0]
            date = elements[1].split('">')[1].split('</a>')[0]
            dict = {"id":logInfo, "date": date, "type":elements[4],
            "airframe":elements[5], "hardware":elements[6], 
            "software":elements[7], "duration":elements[8],
            'startTime':elements[9], "errors":elements[11], "flightModes":elements[12]}
            db_entries_list.append(dict)

    with open('MetaLogs.json', 'w') as outfile:
        json.dump(db_entries_list, outfile)
    return db_entries_list

def loadMetaData():
    with open('MetaLogs.json', 'r') as inputFile:
        return json.load(inputFile)

def main():
  download_folder = 'dataDownloadedHex'
  overwrite = False
  download_api = 'https://review.px4.io/download'
  db_entries_list = browseData()
  #db_entries_list = loadMetaData()

  if not os.path.isdir(download_folder): # returns true if path is an existing directory
            print("creating download directory " + download_folder)
            os.makedirs(download_folder)
  # find already existing logs in download folder
  logfile_pattern = os.path.join(os.path.abspath(download_folder), "*.ulg")
  logfiles = glob.glob(os.path.join(os.getcwd(), logfile_pattern))
  logids = frozenset(os.path.splitext(os.path.basename(f))[0] for f in logfiles)
  #types = ['Quadrotor', 'Fixed Wing']
  types = ['Hexarotor']

  db_entries_list = [entry for entry in db_entries_list
                        if entry["type"] in types]
  n_en = len(db_entries_list)
  print(n_en)
  n_downloaded = 0
  n_skipped = 0
  for i in range(n_en):
    entry_id = db_entries_list[i]['id']
    num_tries = 0
    for num_tries in range(100):
        try:
            if overwrite or entry_id not in logids:

                file_path = os.path.join(download_folder, entry_id + ".ulg")

                print('downloading {:}/{:} ({:})'.format(i + 1, n_en, entry_id))
                request = requests.get(url=download_api +
                                        "?log=" + entry_id, stream=True)
                with open(file_path, 'wb') as log_file:
                    for chunk in request.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            log_file.write(chunk)
                n_downloaded += 1
            else:
                n_skipped += 1
            break
        except Exception as ex:
            print(ex)
            print('Waiting for 30 seconds to retry')
            time.sleep(30)
    if num_tries == 99:
        print('Retried', str(num_tries + 1), 'times without success, exiting.')
        sys.exit(1)


  print('{:} logs downloaded to {:}, {:} logs skipped (already downloaded)'.format(
    n_downloaded, download_folder, n_skipped))         
if __name__ == '__main__':
    main()

URL = "https://review.px4.io/browse_data_retrieval?draw=1 \
          &columns%5B0%5D%5Bdata%5D=0&columns%5B0%5D%5Bname%5D=&columns%5B0%5D%5Bsearchable%5D=true \
          &columns%5B0%5D%5Borderable%5D=false \
          &columns%5B0%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B0%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B1%5D%5Bdata%5D=1&columns%5B1%5D%5Bname%5D=&columns%5B1%5D%5Bsearchable%5D=true \
          &columns%5B1%5D%5Borderable%5D=true&columns%5B1%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B1%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B2%5D%5Bdata%5D=2&columns%5B2%5D%5Bname%5D=&columns%5B2%5D%5Bsearchable%5D=true\
          &columns%5B2%5D%5Borderable%5D=false \
          &columns%5B2%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B2%5D%5Bsearch%5D%5Bregex%5D=false\
          &columns%5B3%5D%5Bdata%5D=3&columns%5B3%5D%5Bname%5D=&columns%5B3%5D%5Bsearchable%5D=true\
          &columns%5B3%5D%5Borderable%5D=true \
          &columns%5B3%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B3%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B4%5D%5Bdata%5D=4&columns%5B4%5D%5Bname%5D=&columns%5B4%5D%5Bsearchable%5D=true  \
          &columns%5B4%5D%5Borderable%5D=true \
          &columns%5B4%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B4%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B5%5D%5Bdata%5D=5&columns%5B5%5D%5Bname%5D=&columns%5B5%5D%5Bsearchable%5D=true \
          &columns%5B5%5D%5Borderable%5D=false \
          &columns%5B5%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B5%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B6%5D%5Bdata%5D=6&columns%5B6%5D%5Bname%5D=&columns%5B6%5D%5Bsearchable%5D=true \
          &columns%5B6%5D%5Borderable%5D=true \
          &columns%5B6%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B6%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B7%5D%5Bdata%5D=7&columns%5B7%5D%5Bname%5D=&columns%5B7%5D%5Bsearchable%5D=true \
          &columns%5B7%5D%5Borderable%5D=true \
          &columns%5B7%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B7%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B8%5D%5Bdata%5D=8&columns%5B8%5D%5Bname%5D=&columns%5B8%5D%5Bsearchable%5D=true \
          &columns%5B8%5D%5Borderable%5D=true \
          &columns%5B8%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B8%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B9%5D%5Bdata%5D=9&columns%5B9%5D%5Bname%5D=&columns%5B9%5D%5Bsearchable%5D=true\
          &columns%5B9%5D%5Borderable%5D=true \
          &columns%5B9%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B9%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B10%5D%5Bdata%5D=10&columns%5B10%5D%5Bname%5D=&columns%5B10%5D%5Bsearchable%5D=true \
          &columns%5B10%5D%5Borderable%5D=false \
          &columns%5B10%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B10%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B11%5D%5Bdata%5D=11&columns%5B11%5D%5Bname%5D=&columns%5B11%5D%5Bsearchable%5D=true \
          &columns%5B11%5D%5Borderable%5D=true \
          &columns%5B11%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B11%5D%5Bsearch%5D%5Bregex%5D=false \
          &columns%5B12%5D%5Bdata%5D=12&columns%5B12%5D%5Bname%5D=&columns%5B12%5D%5Bsearchable%5D=true \
          &columns%5B12%5D%5Borderable%5D=false \
          &columns%5B12%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B12%5D%5Bsearch%5D%5Bregex%5D=false \
          &order%5B0%5D%5Bcolumn%5D=1&order%5B0%5D%5Bdir%5D=desc\
          &start=%d&length=%d&search%5Bvalue%5D=&search%5Bregex%5D=false&_=1644687635985"
