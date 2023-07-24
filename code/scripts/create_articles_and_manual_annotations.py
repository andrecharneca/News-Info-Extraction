'''
    This script does 3 things:
    1. Does some filtering on the raw manual annotations (Business news articles - Manual Articles.csv), like changing column names and removing bad websites
    2. Downloads all good articles into jsons in ARTICLES_JSONS_PATH
    3. Saves the filtered manual annotations into MANUAL_ANNOTATIONS_DF_PATH

    Note: CNBC articles have to be manually transcribed, bc the text stops at the first header.
'''

import sys
import os
import json
sys.path.append("../src/")
from pipelines.utils.newspaper_utils import get_article
from pipelines.paths import (MANUAL_ANNOTATIONS_RAW_DF_PATH, 
                                ARTICLES_JSONS_PATH,
                                MANUAL_ANNOTATIONS_JSONS_PATH,)
import pandas as pd
from tqdm import tqdm

def get_new_url_list():
    """
    Gets URLS from MANUAL_ANNOTATIONS_RAW_DF_PATH that are not in ARTICLES_JSONS_PATH
    (aka new articles that need to be downloaded)
    """
    # open the folder with article jsons, and get all unique urls
    try:
        old_url_list = [json.load(open(os.path.join(ARTICLES_JSONS_PATH, json_file)))["url"] 
                        for json_file in os.listdir(ARTICLES_JSONS_PATH) 
                        if json_file.endswith(".json")]
    except FileNotFoundError:
        old_url_list = []

    df = pd.read_csv(MANUAL_ANNOTATIONS_RAW_DF_PATH)
    new_url_list = df["URL"].unique().tolist()
    new_url_list = [url for url in new_url_list if url not in old_url_list]
    return new_url_list

def download_articles_from_list(url_list):
    df = pd.read_csv(MANUAL_ANNOTATIONS_RAW_DF_PATH)

    good_articles = []
    for url in tqdm(url_list):
        article = get_article(url, verbose=True)
        # df_articles = pd.concat([df_articles, pd.DataFrame([{"article_id":df[df["URL"]==url]["Article ID"].values[0],
        #     "url": url, "title": article["title"], "text": article["text"], "date": article["date"]}])], ignore_index=True)

        # filter out bad articles
        if article["text"] != "" or \
            article["text"] != " " or \
            (url.contains("cnn.com") and article["text"].endswith("Read More")) or \
            url.contains("bbc.com"):
            article_id = int(df[df["URL"]==url]["Article ID"].values[0])
            date = article["date"].strftime("%Y-%m-%d %H:%M:%S") if article["date"] else None
            good_articles.append({"article_id":article_id,
            "url": url, "title": article["title"], "text": article["text"], "date": date})

    # save good articles to jsons
    article_file_name = "article_id_{article_id}.json"
    for article in good_articles:
        with open(os.path.join(ARTICLES_JSONS_PATH, article_file_name.format(article_id=article["article_id"])), "w") as f:
            json.dump(article, f, indent=4)

def create_manual_annotations():
    df = pd.read_csv(MANUAL_ANNOTATIONS_RAW_DF_PATH)[['Article ID', 'URL', 'Entity_1', 'Relationship', 'Entity_2', 'Entities', 'Relevant Passage']]
    
    # for each unique url, create a dict with keys: article_id, paragraph_id(=0), url, relationships.
    # relationships is a list of dicts with keys: entity_1, relationship, entity_2, passage;
    #    that contains all relationships of that url
    article_json_name = "article_id_{article_id}.json"
    
    for url in tqdm(df["URL"].unique().tolist()):
        df_url = df[df["URL"]==url]
        article_id = int(df_url["Article ID"].values[0])
        
        entities = df_url["Entities"].values[0]
        if entities != entities: # if entities is NaN
            entity_list = []
        else:
            words = entities.split(",")
            words_list = [word.strip() for word in words]  
            entity_list = [{'word': word} for word in words_list]      

        # load article json
        article = json.load(open(os.path.join(ARTICLES_JSONS_PATH, article_json_name.format(article_id=article_id))))
        relationships = []

        for i, row in df_url.iterrows():
            # if things are NaN, skip
            if pd.isna(row["Entity_1"]) or pd.isna(row["Relationship"]) or pd.isna(row["Entity_2"]):
                continue
            else:
                relationships.append({"entity_1": row["Entity_1"], "relationship": row["Relationship"], "entity_2": row["Entity_2"], "passage": row["Relevant Passage"]})
            
        article = {"url": url,"article_id": article_id, "title": article["title"], "text": article["text"],
                    "relationships": relationships, "entities": entity_list}

        with open(os.path.join(MANUAL_ANNOTATIONS_JSONS_PATH, article_json_name.format(article_id=article_id)), "w") as f:
            json.dump(article, f, indent=4)
    
if __name__ == "__main__":
    new_url_list = get_new_url_list()
    print("New articles to download: ", new_url_list)
    download_articles_from_list(new_url_list)
    #create_manual_annotations()



