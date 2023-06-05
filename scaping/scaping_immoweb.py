import requests, json, lxml, re, csv
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import pandas as pd
from typing import List, Dict
import itertools
from functools import partial
from tqdm.contrib.concurrent import thread_map

def get_property_urls_from_search_page(search_url):
    """
    Fetches URLs of individual properties from a given search page.
    """
    response = requests.get(search_url)
    webpage_content = BeautifulSoup(response.content, 'html.parser')
    property_cards = webpage_content.find_all('article', class_='card--result')
    
    property_links = []
    for property_card in property_cards:
        link = property_card.find('a', class_='card__title-link')
        if link:
            property_links.append(link['href'])
    return property_links

def generate_search_urls(end_page):
    """
    Generates search page URLs for different property types.
    """
    property_types = ["house","apartment"]

    search_urls = []
    for property_type in property_types:
        for page in range(1, end_page):
            url = f"https://www.immoweb.be/en/search/{property_type}/for-sale?countries=BE&page={page}&orderBy=relevance"
            search_urls.append(url)
    return search_urls

def fetch_property_data(property_url, web_session): 
    """
    Fetches data for an individual property and returns it as a DataFrame.
    """
    try:
        response = web_session.get(property_url)
        html_tables = pd.read_html(response.text)
        property_table = pd.concat(html_tables).set_index(0).T
        property_table["id"] = property_url.split("/")[-1]
        property_table = property_table.set_index("id")
        property_table = property_table.loc[:, ~property_table.columns.duplicated()].copy()

        property_info = []
        window_data = re.findall("window.dataLayer =(.+?);\n", response.text, re.S)
        if window_data:
            property_info.append(json.loads(window_data[0])[0]['classified'])
        
        combined_dict = property_info[0]

        for index in property_table.index:
            combined_dict[property_table[0][index]] = property_table[1][index]

        combined_data_frame = pd.DataFrame([combined_dict])
        
        return combined_data_frame

    except Exception as error:
        print(f"Error: {str(error)} occurred while processing property URL: {property_url}")
        return None

if __name__ == "__main__":
    search_urls = generate_search_urls(end_page=334)

    property_urls = list(itertools.chain.from_iterable(thread_map(get_property_urls_from_search_page, search_urls)))

    with requests.Session() as web_session:
        property_data_frames = [data_frame for data_frame in thread_map(partial(fetch_property_data, web_session=web_session), property_urls) 
                               if data_frame is not None]

    if property_data_frames:  # Check if the list is not empty
        property_data_frames = pd.concat(property_data_frames)
        property_data_frames.to_csv("Data/raw_data.csv")
    else:
        print("No data to write to CSV. List of property data is empty.")

