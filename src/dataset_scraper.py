import io
import pathlib
import requests

import pandas as pd

from bs4 import BeautifulSoup
from zipfile import ZipFile


DATASET_LIST_URL = 'https://webrobots.io/kickstarter-datasets/'
CACHE_PATH = pathlib.Path(__file__).parent.joinpath('../data').resolve()

pathlib.Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)


def get_links():
    """
        Scrapes the links from the DATASET_LIST_URL and
        returns a list of them.
    """

    res = requests.get(DATASET_LIST_URL)

    soup = BeautifulSoup(res.content, 'html.parser')
    links = soup.find_all('a', text='CSV')

    return [link['href'] for link in links]


def test_zipfile(zip_path):
    """
        Returns True if the zip_path is a valid zipfile.
    """

    # Test to see if the file exists
    if not pathlib.Path(zip_path).is_file():
        return False

    try:
        # Test to see if the file fails to open as a zipfile
        with open(zip_path, 'rb') as file:
            ZipFile(io.BytesIO(file.read()), mode='r')
    except Exception:
        return False

    return True


def csv_generator():
    """
        Yields CSVs extracted from the DATASET_LIST_URL.
    """

    links = get_links()
    for link in links:
        filename = link.split("/")[-1]

        filepath = pathlib.Path(CACHE_PATH).joinpath(filename).resolve()

        if not test_zipfile(filepath):
            print(f'Downloading {filename}...')
            res = requests.get(link, stream=True)

            if not res.ok:
                raise Exception(f'Error while trying to download {link}')

            # cache the zipfile
            with open(filepath, 'wb') as file:
                file.write(res.content)
        else:
            print(f'Using cached {filename}...')

        # read the zipfile into memory
        with open(filepath, 'rb') as file:
            zipf = ZipFile(io.BytesIO(file.read()), mode='r')
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
