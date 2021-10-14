import io
import requests

import pandas as pd

from bs4 import BeautifulSoup
from zipfile import ZipFile


DATASET_LIST_URL = 'https://webrobots.io/kickstarter-datasets/'


def get_links():
    """
        Scrapes the links from the DATASET_LIST_URL and
        returns a list of them.
    """

    res = requests.get(DATASET_LIST_URL)

    soup = BeautifulSoup(res.content, 'html.parser')
    links = soup.find_all('a', text='CSV')

    return [link['href'] for link in links]


def csv_generator():
    """
        Yields CSVs extracted from the DATASET_LIST_URL.
    """

    links = get_links()
    for link in links:
        filename = link.split("/")[-1]

        print(f'Downloading {filename}...')
        res = requests.get(link, stream=True)

        if not res.ok:
            raise Exception(f'Error while trying to download {link}')

        # read the zipfile into memory
        zipf = ZipFile(io.BytesIO(res.content))
        for csv in zipf.namelist():
            print(f'\tExtracting {csv}...')
            # extract the csv from the zipfile in memory
            # and decode it as utf8
            yield zipf.read(csv).decode('utf8')


def dataframe_generator():
    """
        Yields dataframes extracted from the DATASET_LIST_URL.

        Example:
        ```
            from dataset_scraper import dataframe_generator

            for df in dataframe_generator():
                print(df.head())
        ```
    """

    for csv in csv_generator():
        # convert the string into StringIO
        # to make pandas happy.
        inp = io.StringIO(csv)
        yield pd.read_csv(inp)
