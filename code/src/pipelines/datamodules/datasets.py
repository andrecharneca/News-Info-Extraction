from pprint import pprint
import pandas as pd
from typing import Union
import numpy as np
import json
import os

from transformers import AutoTokenizer
from torch.utils.data import Dataset

from pipelines.paths import ARTICLES_JSONS_PATH
from pipelines.datamodules.utils import ParagraphUtils

# TODO: check if the keys of all the dicts are the same
class ParagraphDataset(Dataset):
    """
    Gets paragraphs from articles in a dataframe, outputs dicts
        - data_path must be a path to a dataframe with columns "url", "title", "text"
        - tokenizer is needed to concatenate paragraphs
    """
    def __init__(self, data_path : str = ARTICLES_JSONS_PATH, 
                 data : list[dict] = None,
                 article_id_list : list[int] = None,
                 tokenizer_model_name : str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", 
                 max_tokens : int = 256,
                 include_title_in_text : bool = True):
        super().__init__()

        self.data_path = data_path
        self.tokenizer_model_name = tokenizer_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.max_tokens = max_tokens
        self.include_title_in_text = include_title_in_text
        self.data = []
        self.article_id_list = article_id_list

        if data is not None:
            self.data = data
            self._select_articles()
        else:
            self._load_data()

    def _load_data(self):
        if self.article_id_list is None:
            # load all articles
            articles = []
            for filename in os.listdir(self.data_path):
                if filename.endswith(".json"):
                    articles.append(json.load(open(os.path.join(self.data_path, filename))))
        else:
            # load only articles with article_id in article_id_list
            articles = []
            for article_id in self.article_id_list:
                filename = f"article_id_{article_id}.json"
                articles.append(json.load(open(os.path.join(self.data_path, filename))))

        for article in articles:
            article_id, url, text, title = article["article_id"], article["url"], article["text"], article["title"]
            full_text = f"{title} {text}" if self.include_title_in_text else f"{text}"
            paragraphs = ParagraphUtils.text_to_paragraphs(full_text)
            paragraphs = ParagraphUtils.filter_paragraphs(paragraphs)
            paragraphs = ParagraphUtils.concat_paragraphs(self.tokenizer, paragraphs, self.max_tokens)
            
            # each paragraph goes into a dict with keys: url, paragraph_id, text
            self.data.extend([{"article_id": article_id, "url": url, "title":title, "paragraph_id": i, "text": paragraph} for i, paragraph in enumerate(paragraphs)])
    
    def _select_articles(self):
        """
        Selects only the articles with article_id in article_id_list.
        """
        if self.article_id_list is not None:
            self.data = [paragraph for paragraph in self.data if paragraph["article_id"] in self.article_id_list]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def zsc_collate_fn(self, batch):
        """
        Collate function for the torch dataloader.
        dict of lists -> list of dicts
        """
        # for some reason, this is enough to convert the dict of lists to a list of dicts
        return batch
    
    @property
    def keys(self):
        return self.data[0].keys()
    
    def get_key_vals(self, key : str):
        return list(np.unique([p[key] for p in self.data]))

    def get_filtered_by_key_vals(self, key : str, vals):
        """
        Returns a list of paragraphs where the value of the key is in vals.
        """
        if not isinstance(vals, list):
            vals = [vals]
        return [p for p in self.data if p[key] in vals]
    
    def get_paragraph(self, article_id : int, paragraph_id : int):
        """
        Returns the paragraph with article_id and paragraph_id.
        """
        return [p for p in self.data if p["article_id"] == article_id and p["paragraph_id"] == paragraph_id][0]
    
    def set_data(self, data : list[dict]):
        self.data = data

    def remove_key_from_data(self, key : str):
        """
        Removes a key from the dataset.
        """
        self.data = [{k: v for k, v in paragraph.items() if k != key} for paragraph in self.data]
        
    def add_key_vals(self, vals : list, key : str):
        """
        Adds a new key to the dataset with the values in vals. 
        They must be in same order.
        """
        assert len(vals) == len(self.data)
        self.data = [{**paragraph, key: vals[i]} for i, paragraph in enumerate(self.data)]

    def to_jsons(self, folder_path : str):
        """
        Saves each paragraph as a json file in the folder_path.
        """
        is_article_dataset = "paragraph_id" not in self.keys

        # make folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        json_name = "article_id_{article_id}.json" if is_article_dataset else "article_id_{article_id}_paragraph_id_{paragraph_id}.json"

        for paragraph in self.data:
            if is_article_dataset:
                paragraph_id = 0
            else:
                paragraph_id = paragraph["paragraph_id"]

            file_name = json_name.format(article_id=paragraph["article_id"], paragraph_id=paragraph_id)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "w") as f:
                json.dump(paragraph, f, indent=4)


    def to_df(self):
        """
        Converts the dataset to a pandas dataframe.
        Note: Expand the relationships first, with expand_relationships().
        """
        # empty dataframe
        df = pd.DataFrame()
        # all the keys in the dataset, except "relationships" and "entities"
        paragraph_keys = [key for key in self.keys if key not in ["relationships", "entities"]]

        for paragraph in self.data:
            if len(paragraph["relationships"]) > 0:
                row = {key: paragraph[key] for key in paragraph_keys}
                row.update({key: paragraph["relationships"][0][key] for key in paragraph["relationships"][0].keys()})
                
                df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

        return df
    
    def to_csv(self, path : str):
        """
        Saves the dataset to a csv file.
        To have one relationship per row, use expand_relationships() first.

        Saves to df with columns: url, paragraph_id, text, [relationship keys]
        """
        df = self.to_df()
        df.to_csv(path, index=False)

    