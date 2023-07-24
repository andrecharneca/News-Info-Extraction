from typing import Union
import csv
import os
import json
from ..datamodules.datasets import ParagraphDataset
from ..datamodules.utils import RelationshipUtils
import re
from .entities import ENTITY_STOPWORDS, NEWS_SITE_TO_ENTITIES
from .relationships import BIDIR_RELATIONSHIP_LIST
from difflib import SequenceMatcher
from collections import defaultdict
import numpy as np
import copy
import parse
import cleanco

class GPT3Utils(object):
    @staticmethod
    def parse_table(completion: str):
        """
        Turns the text from the openai gpt response into a list of table rows
        args:
            completion: completion from openai, in the format: |Company|Relationship|Company|Passage|
            ending: ending of the table (use 'end' if the table inds with '|end|')
        returns:
            list: list of relationships, like [{"entity_1":'Apple', "entity_2":'iPhone', "relationship":'owner of', "passage":'Apple is the owner of iPhone']
        """
        # remove the first row (the prompt)
        rows = [word.fixed for word in parse.findall("|{}|{}|{}|{}|", completion)][1:]
        relationships = [{"entity_1":cleanco.basename(row[0]), "relationship":row[1], "entity_2":cleanco.basename(row[2]), "passage":row[3]} for row in rows]
        return relationships

    
    def parse_entity_list(completion: str):
        """
        Parses completions of the form [entity_1, entity_2, entity_3, ...]
        """
        # parse entities
        entities = []
        entity_list_text = ''.join(r[0] for r in parse.findall("[{}]", completion))
        for word in entity_list_text.split(","):
            #remove whitespace
            word = word.strip()
            if word:
                entity_group = [r[0] for r in parse.findall("({})", word)][-1]
                entity = word.replace(f"({entity_group})", "").strip()
                entities.append({"word": cleanco.basename(entity), "entity_group": entity_group})

        return entities

    @staticmethod
    def remove_hallucinated_relationships(relationships: list[dict], allowed_relaFtionships: list[str]):
        """
        Removes relationships that are not in allowed_relationships
        """
        return [relationship for relationship in relationships if relationship["relationship"] in allowed_relationships]

    @staticmethod
    def filter_relationships_by_text(relationships: list[dict], text: str) -> list[dict]:
        """
        Filters relationships to only include those where both entities and the passage are present in the text
        TODO: implement passage filtering
        """
        filtered_relationships = []
        for relationship in relationships:
            entity_1 = relationship["entity_1"]
            entity_2 = relationship["entity_2"]
            if entity_1 in text and entity_2 in text:
                filtered_relationships.append(relationship)
        return filtered_relationships

    @staticmethod
    def filter_entities_by_text(entities: list[dict], text: str) -> list[dict]:
        """
        Filters entities to only include those that are present in the text
        """
        filtered_entities = []
        for entity in entities:
            if entity["word"] in text:
                filtered_entities.append(entity)
        return filtered_entities

class ZeroShotUtils(object):
    
    ## Pre zero shot classification ##
    
    @staticmethod
    def compute_relationships_and_add(dataset: ParagraphDataset, **kwargs):
        """
        With the entities in the paragraph, add the 'relationships' key to the paragraph dict.
            - "relationships": list of dicts with keys: "entity_1", "entity_2", "relationship", "sentence"
            - only works if key "entities" is in the paragraph dict
        """
        if "entities" not in dataset.keys:
            raise Exception("Key 'entities' not found in dataset")
        
        dataset.set_data([{**paragraph, "relationships": RelationshipUtils.get_relationships(paragraph["entities"], **kwargs)} for paragraph in dataset.data])

    @staticmethod
    def compute_entity_filtering_relationships_and_add(dataset: ParagraphDataset):
        """
        For filtering NER entities using the Zero Shot Classifier.
        For each entity in the paragraph, adds the 'relationships' key to the paragraph.
        This only contains relations "[ORG] is a company", "[PROD] is a product."
        """
        if "entities" not in dataset.keys:
            raise Exception("Key 'entities' not found in dataset")
        
        dataset.set_data([{**paragraph, "relationships": RelationshipUtils.get_entity_filtering_relationships(paragraph["entities"])} for paragraph in dataset.data])

    @staticmethod
    def filter_entities_by_relationship_score(dataset : ParagraphDataset, threshold : float = 0.99):
        """
        Filters entities by the relationship score.
        This is supposed to be ran after compute_entity_filtering_relationships_and_add.
        """
        new_data = []

        for paragraph in dataset.data:
            # first filter entities by score in the relationships
            keep_entities = []
            for relationship in paragraph["relationships"]:
                if relationship["score"] >= threshold:
                    keep_entities.append(relationship["entity_word"])

            # then keep only entities with "word" matching the keep_entities
            new_paragraph = {**paragraph, "entities": [entity for entity in paragraph["entities"] if entity["word"] in keep_entities]}
            new_data.append(new_paragraph)

        dataset.set_data(new_data)
            

    ## Post zero shot classification ##

    @staticmethod
    def add_scores(dataset: ParagraphDataset, scores : list):
        """
        Adds the scores to the dataset.
        Only works correctly if each paragraph only has one relationship (expanded) in this way:
            "relationships": List[Dict] with only one element
        """
        assert len(scores) == len(dataset.data)
        dataset.set_data([{**paragraph, "relationships": [{**paragraph["relationships"][0], "score": scores[i]}]} for i, paragraph in enumerate(dataset.data)])

    @staticmethod
    def filter_relationships_by_score(dataset: ParagraphDataset, threshold : float = 0.95):
        """
        Removes relationships with scores below threshold
        """
        dataset.set_data([{**paragraph, "relationships": [relationship for relationship in paragraph["relationships"] 
                                                          if relationship["score"] >= threshold]} for paragraph in dataset.data])

    @staticmethod
    def keep_highest_score_duplicate_rels(dataset: ParagraphDataset):
        """
        Keeps only the highest scoring relationship for each set of (entity_1, entity_2), 
        regardless of the order of the entities, for a certain article.

        For example: Google and Apple are competitors, score 0.9. Apple and Google are partners, score 0.8.
        Keep only the relationship "Google and Apple are competitors", since it has the highest score.

        Note: to be used in article-wise datasets, use to_article_dataset first. 
            "relationships" must have the key "scores"
        """
        new_data = []
        for article in dataset.data:
            entity_pairs = set()
            relationships = article["relationships"]

            # first we get all the entity pairs
            for relationship in relationships:
                entity_1 = relationship["entity_1"]
                entity_2 = relationship["entity_2"]
                # we do this with sets so the entity order doesn't matter
                entity_pairs.add(frozenset([entity_1, entity_2]))

            # then we get the highest scoring relationship for each entity pair
            highest_scoring_relationships = []
            for entity_pair in entity_pairs:
                highest_scoring_relationship = max([relationship for relationship in relationships if frozenset([relationship["entity_1"], relationship["entity_2"]]) == entity_pair], key=lambda relationship: np.mean(relationship["scores"]))
                highest_scoring_relationships.append(highest_scoring_relationship)

            article["relationships"] = highest_scoring_relationships
            new_data.append(article)

        dataset.set_data(new_data)
                                                

class DatasetUtils(object):
    @staticmethod
    def expand_relationships(dataset: Union[ParagraphDataset, list[dict]]):
        """
        Expands each paragraph into various paragraphs, one for each relationship
        This enables efficient batching for zero shot classification
        """
        if isinstance(dataset, ParagraphDataset):
            dataset_data = dataset.data
        else:
            dataset_data = dataset
            
        expanded_data = []
        for paragraph in dataset_data:
            relationships = paragraph["relationships"]
            # if there are no relationships, just add the paragraph
            if len(relationships) == 0:
                expanded_data.append(paragraph)
                continue
            for relationship in relationships:
                new_paragraph = {**paragraph, "relationships": [relationship]}
                expanded_data.append(new_paragraph)

        if isinstance(dataset, ParagraphDataset):
            dataset.set_data(expanded_data)
        else:
            return expanded_data

    @staticmethod
    def remove_paragraphs_without_relationships(dataset: ParagraphDataset):
        """
        Removes paragraphs without relationships
        NOTE: This could remove whole articles if there are no relationships in the article.
        """
        dataset.set_data([paragraph for paragraph in dataset.data if len(paragraph["relationships"]) > 0])

            
    @staticmethod
    def contract_relationships(dataset: ParagraphDataset):
        """
        Does the opposite of expand_relationships, joins paragraphs by matching "url" and "paragraph_id"
        """
        contracted_data = []
        for url,id in set([(paragraph["url"], paragraph["paragraph_id"]) for paragraph in dataset.data]):
            paragraphs = [paragraph for paragraph in dataset.data if paragraph["url"] == url and paragraph["paragraph_id"] == id]
            relationships = sum([paragraph["relationships"] for paragraph in paragraphs], [])
            new_paragraph = {**paragraphs[0], "relationships": relationships}
            contracted_data.append(new_paragraph)

        dataset.set_data(contracted_data)    
    
    @staticmethod
    def order_bidir_relationships_entities(dataset: ParagraphDataset, bidir_relationship_list : list[str] = BIDIR_RELATIONSHIP_LIST):
        """
        For bidirectional relationships, orders the entities alphabetically.
        """
        new_data = []
        for paragraph in dataset.data:
            for relationship in paragraph["relationships"]:
                if "relationship" in relationship and relationship["relationship"] in bidir_relationship_list:
                    relationship["entity_1"], relationship["entity_2"] = sorted([relationship["entity_1"], relationship["entity_2"]])
            new_data.append(paragraph)
        dataset.set_data(new_data)


    @staticmethod
    def to_article_dataset(dataset: ParagraphDataset) -> ParagraphDataset:
        """
        Joins paragraphs' entities and relationships into articles.

        Ex: from
        [article_id:0, paragraph_d:0, entities: [{A, score:0.8}, {B, score:0.8}], relationships: [A and B are partners, score:0.99]]
        to
        [article_id:0, 
            entities:[{A,paragraph_ids:[0], scores: [0.8]}, {B,paragraph_ids:[0], scores: [0.8]}], 
            relationships:[{A and B are partners, paragraph_ids:[0], scores: [0.99]}]]
        """
        new_data = []
        for article_id in dataset.get_key_vals("article_id"):
            article_paragraphs = [paragraph for paragraph in dataset.data if paragraph["article_id"] == article_id]
            
            # get article text from article json
            article = json.load(open(os.path.join(dataset.data_path, f"article_id_{article_id}.json")))
            
            # entities
            if "entities" in dataset.keys:
                entities = []
                for paragraph in article_paragraphs:
                    for entity in paragraph["entities"]:
                        # if 'word' and 'entity_group' matches another entity, we add the paragraph_id to that entity
                        if any([entity["word"] == existing_entity["word"] and entity["entity_group"] == existing_entity["entity_group"] for existing_entity in entities]):
                            for existing_entity in entities:
                                if entity["word"] == existing_entity["word"] and entity["entity_group"] == existing_entity["entity_group"]:
                                    existing_entity["paragraph_ids"].append(paragraph["paragraph_id"])
                                    if "score" in entity.keys():
                                        existing_entity["scores"].append(float(entity["score"]))
                                    break
                        # otherwise we add the entity to the list of entities
                        else:
                            if "score" in entity.keys():
                                entity["scores"] = [float(entity["score"])]
                                entity.pop("score")
                            entity["paragraph_ids"] = [paragraph["paragraph_id"]]
                            entities.append(entity)

            # relationships
            if "relationships" in dataset.keys:
                relationships = []
                for paragraph in article_paragraphs:
                    for relationship in paragraph["relationships"]:
                        # if 'entity_1', 'entity_2', and 'relationship' matches another relationship, we add the paragraph_id to that relationship
                        triple = (relationship["entity_1"], relationship["entity_2"], relationship["relationship"])
                        if any([triple == (existing_relationship["entity_1"], existing_relationship["entity_2"], existing_relationship["relationship"]) for existing_relationship in relationships]):
                            for existing_relationship in relationships:
                                if triple == (existing_relationship["entity_1"], existing_relationship["entity_2"], existing_relationship["relationship"]):
                                    existing_relationship["paragraph_ids"].append(paragraph["paragraph_id"])
                                    if "score" in relationship:
                                        existing_relationship["scores"].append(float(relationship["score"]))
                                    break
                        # otherwise we add the relationship to the list of relationships
                        else:
                            if "score" in relationship:
                                relationship["scores"] = [float(relationship["score"])]
                                relationship.pop("score")
                            relationship["paragraph_ids"] = [paragraph["paragraph_id"]]
                            relationships.append(relationship)

            new_article = {
                "article_id": int(article_id),
                "url": article["url"],
                "title": article["title"],
                "text": article["text"],
            }

            if "entities" in dataset.keys:
                new_article["entities"] = entities
            if "relationships" in dataset.keys:
                new_article["relationships"] = relationships

            new_data.append(new_article)
        dataset.set_data(new_data)
    
    @staticmethod
    def keep_relationships(dataset: ParagraphDataset, relationships: list[str]) -> ParagraphDataset:
        """
        Filters the dataset to only include relationships with the given relationship type.
        """
        result = copy.deepcopy(dataset)
        result.set_data([{**paragraph, "relationships": [relationship for relationship in paragraph["relationships"] if relationship["relationship"] in relationships]} for paragraph in dataset.data])
        return result

    @staticmethod
    def load_csv(path: str, 
                 relationship_keys: list[str] = ["entity_1", "entity_2", "relationship", "passage"],
                 paragraph_keys: list[str] = ["article_id", "url", "title", "paragraph_id", "text"],
                 **kwargs):
        """
        Loads a csv file into a ParagraphDataset
        
        Assumes the csv has 1 relationship per row.
        
        For ZSC output, relationship_keys should be ["entity_1", "entity_2", "relationship", "score", "sentence"(optional))]
        For GPT output, relationship_keys should be ["entity_1", "entity_2", "relationship", "passage"]
        For manual annotations, relationship_keys should be ["entity_1", "entity_2", "relationship", "passage"]

        These keys are stored in a list[dict] called "relationships" in each paragraph
        The other keys are stored in the paragraph dict
        """
        
        #convert csv to list of dicts, one dict per row
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = []
            for row in reader:
                relationships = [{key: row[key] for key in relationship_keys}]
                paragraph = {key: row[key] for key in paragraph_keys}
                paragraph["relationships"] = relationships
                data.append(paragraph)         

        # convert to ParagraphDataset
        dataset = ParagraphDataset(data=data, **kwargs)
        return dataset
    
    @staticmethod
    def load_jsons(folder: str, **kwargs):
        """
        Loads a folder of jsons into a ParagraphDataset.
        Each json should be a paragraph with the name "article_id_[]_paragraph_id_[].json"
        """
        data = []
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                with open(os.path.join(folder, filename)) as f:
                    data.append(json.load(f))
        dataset = ParagraphDataset(data=data, **kwargs)
        return dataset
    
        
class MetricsUtils(object):
    """
    For the special case of 0s in FP, FN, TP, TN, see https://stats.stackexchange.com/a/305905
    """
    @staticmethod
    def get_rel_sentences_from_paragraphs(paragraphs: list[dict]):
        """
        To simplify the compute_precision
        """
        paragraph_rels = [rel for paragraph in paragraphs for rel in paragraph["relationships"]]

        # if relationships dont have sentences, create them
        if len(paragraph_rels) == 0:
            return []
        if "sentence" not in paragraph_rels[0].keys():
            paragraph_triples = [{"entity_1": rel["entity_1"], "entity_2": rel["entity_2"], "relationship": rel["relationship"]}
                                   for rel in paragraph_rels]

            dataset_sentences =  RelationshipUtils.triples_to_sentences(paragraph_triples)
        else:
            dataset_sentences = [rel["sentence"] for rel in paragraph_rels]

        return dataset_sentences
    
    @staticmethod
    def compute_intersection_relationships(predictions: ParagraphDataset, targets: ParagraphDataset):
        '''
        Computes the intersection of the predicted and target relationships.
        dataset of dicts with keys: "relationships", "article_id"
        "relationships" is a list of dicts with keys: "entity_1", "entity_2", "relationship"
        '''
        predictions_rels = set()
        for paragraph in predictions.data:
            for relationship in paragraph["relationships"]:
                rel = (relationship["entity_1"].lower(), relationship["entity_2"].lower(), relationship["relationship"].lower(), paragraph["article_id"])
                predictions_rels.add(rel)

        targets_rels = set()
        for paragraph in targets.data:
            for relationship in paragraph["relationships"]:
                rel = (relationship["entity_1"].lower(), relationship["entity_2"].lower(), relationship["relationship"].lower(), paragraph["article_id"])
                targets_rels.add(rel)
        intersection = predictions_rels.intersection(targets_rels)
        return intersection
    
    @staticmethod
    def compute_intersection_entities(predictions: ParagraphDataset, targets: ParagraphDataset):
        '''
        Computes the intersection of the predicted and target entities.
        dataset of dicts with keys: "entities", "article_id"
        "entities" is a list of dicts with key: "word"
        '''
        predictions_entities = set()
        for paragraph in predictions.data:
            for entity in paragraph["entities"]:
                predictions_entities.add((entity["word"].lower(), paragraph["article_id"]))
        targets_entities = set()
        for paragraph in targets.data:
            for entity in paragraph["entities"]:
                targets_entities.add((entity["word"].lower(), paragraph["article_id"]))
        intersection = predictions_entities.intersection(targets_entities)
        return intersection

    @staticmethod
    def compute_tp_fp_fn(predictions: ParagraphDataset, targets: ParagraphDataset, key="relationships"):
        '''
        Computes the precision, article by article.
        
        Note: before running this, convert to article dataset with DatasetUtils.to_article_dataset(dataset).
            This way, each data point is an article with multiple relationships.
        '''
        if key == "relationships":
            intersection = MetricsUtils.compute_intersection_relationships(predictions, targets)
        elif key == "entities":
            intersection = MetricsUtils.compute_intersection_entities(predictions, targets)
        else:
            raise ValueError("key must be 'relationships' or 'entities'")
        TP = len(intersection)
        TP_plus_FP = len([rel for paragraph in predictions.data for rel in paragraph[key]])
        TP_plus_FN = len([rel for paragraph in targets.data for rel in paragraph[key]])
        FP = TP_plus_FP - TP
        FN = TP_plus_FN - TP
        return TP, FP, FN
    
    @staticmethod
    def compute_metrics(predictions: ParagraphDataset, targets: ParagraphDataset, key="relationships"):
        '''
        Computes the metrics of the predictions.
        '''
        TP, FP, FN = MetricsUtils.compute_tp_fp_fn(predictions, targets, key=key)
        precision = (TP) / (TP+FP+1e-10)
        recall = (TP) / (TP+FN+1e-10)

        if TP == 0 and FP == 0 and FN == 0:
            # if no rels predicted and no rels in target, precision is 1
            precision = 1
            recall = 1
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return precision, recall, f1

    
    @staticmethod
    def compute_avg_article_metrics(predictions: ParagraphDataset, targets: ParagraphDataset):
        """
        Computes the average metrics over the articles.
        """
        all_article_ids = set([paragraph["article_id"] for paragraph in predictions.data]).union(set([paragraph["article_id"] for paragraph in targets.data]))
        metrics = {
            "relationships": {"precision": 0., "recall": 0., "f1": 0.},
            "entities": {"precision": 0., "recall": 0., "f1": 0.}
        }

        for article_id in all_article_ids:
            article_predictions = ParagraphDataset(data=predictions.get_filtered_by_key_vals("article_id", [article_id]))
            article_targets = ParagraphDataset(data=targets.get_filtered_by_key_vals("article_id", [article_id]))
            rel_precision, rel_recall, rel_f1 = MetricsUtils.compute_metrics(article_predictions, article_targets, key="relationships")
            entity_precision, entity_recall, entity_f1 = MetricsUtils.compute_metrics(article_predictions, article_targets, key="entities")
            metrics["relationships"]["precision"] += rel_precision
            metrics["relationships"]["recall"] += rel_recall
            metrics["relationships"]["f1"] += rel_f1
            metrics["entities"]["precision"] += entity_precision
            metrics["entities"]["recall"] += entity_recall
            metrics["entities"]["f1"] += entity_f1

        for metric in metrics:
            for submetric in metrics[metric]:
                metrics[metric][submetric] /= len(all_article_ids)
        return metrics
    
class EntityUtils(object):
    @staticmethod
    def longest_match(a, b):
        return SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    
    @staticmethod
    def clean(company_name : str) -> list[str]:
        result = company_name.lower()
        result = re.sub(r'[^\w\s]','',result) # remove punctuation

        # split on whitespace and remove empty strings
        result = [word for word in result.split(" ") if word != ""]

        # remove stopwords
        result = [word for word in result if word not in ENTITY_STOPWORDS]
        return result
    
    @staticmethod
    def find_shortest_match(entity, target_entities):
        """
        Finds the shortest (in terms of number of words) match between an entity and a list of target entities.

        If there is a mention starting with a capital letter in the target entities,
        the shortest match will be capitalized.

        Args:
            entity (str): The entity to match.
            target_entities (list): A list of target entities to match against.

        Returns:
            str: The shortest match between the entity and the target entities.
        """
        word_matches = []
        for e in target_entities:
            match_ = EntityUtils.longest_match(EntityUtils.clean(entity), EntityUtils.clean(e))
            if match_.size > 0 and match_.a == 0:
                word_matches.append(e)
        shortest_match = min(word_matches, key=len) if len(word_matches) > 0 else entity
        if len(word_matches) > 0 and any([e[0].isupper() for e in word_matches + [entity]]):
            shortest_match = shortest_match[0].upper() + shortest_match[1:]
        return shortest_match
    
    @staticmethod
    def apply_entity_linking(dataset : ParagraphDataset):
        """
        Applies entity linking to a ParagraphDataset.
        Entities are linked at an article level.
            - dataset.data must have keys "article_id", "paragraph_id", "entities"
            - dataset.data["entities"] must be a list of dicts with keys "word", "entity_group"
        
        Modifies the input dataset in place

        Example: 
        {"paragraph_id": 0, "word": "Apple Company"}, {"paragraph_id": 1, "word": "Apple"}
        both get reduced to the smallest entity "Apple"
        """

        new_data = []
        for article_id in dataset.get_key_vals("article_id"):
            #get all entities in article
            article_pars = dataset.get_filtered_by_key_vals("article_id", [article_id])
            article_entities = [{**entity, "paragraph_id": par["paragraph_id"]}
                                 for par in article_pars for entity in par["entities"]]

            # sort entities by length
            article_entities = sorted(article_entities, key=lambda x: len(x["word"]), reverse=True)

            for entity in article_entities:
                same_group_entities = [e for e in article_entities if e["entity_group"] == entity["entity_group"]
                                       if e["word"] != entity["word"]]
                
                # find shortest match
                shortest_match = EntityUtils.find_shortest_match(entity["word"], [e["word"] for e in same_group_entities])
                entity["word"] = shortest_match

            # assign entities to paragraphs
            for par in article_pars:
                par["entities"] = [e for e in article_entities if e["paragraph_id"] == par["paragraph_id"]]

                # remove entities with duplicate words
                par["entities"] = list({e["word"]:e for e in par["entities"]}.values())
                new_data.append(par)
                
        # remove paragraph_id key from entities
        for par in new_data:
            for entity in par["entities"]:
                entity.pop("paragraph_id")

        dataset.set_data(new_data)
    
    @staticmethod
    def link_entities_to_target(predictions: ParagraphDataset, target: ParagraphDataset):
        """
        Links entities from the relationships and entities to the target ones, in a similar way to EntityUtils.apply_entity_linking
        To be used for evaluating GPT-3 predictions.
        Note: use this on article-wise datasets, after using DatasetUtils.to_article_dataset
        """
        for pred in predictions.data:
            article_id = pred["article_id"]
            target_data = target.get_filtered_by_key_vals("article_id", [article_id])[0]
            target_rel_entities = [rel["entity_1"] for rel in target_data["relationships"]] + [rel["entity_2"] for rel in target_data["relationships"]]
            # reduce to unique entities
            target_rel_entities = list(set(target_rel_entities))

            for rel in pred["relationships"]:
                # find shortest matches
                rel["entity_1"] = EntityUtils.find_shortest_match(rel["entity_1"], target_rel_entities)
                rel["entity_2"] = EntityUtils.find_shortest_match(rel["entity_2"], target_rel_entities)

            for entity in pred["entities"]:
                # find shortest match for entity
                entity["word"] = EntityUtils.find_shortest_match(entity["word"], target_rel_entities)



    @staticmethod
    def remove_news_site_entities(dataset: ParagraphDataset, news_site_to_entities: list = NEWS_SITE_TO_ENTITIES):
        """
        Remove entities that are news sites, if the article came from that site.
        """
        for paragraph in dataset.data:
            website = [key for key in news_site_to_entities if key in paragraph["url"]]
            if len(website) > 0:
                website = website[0]
                entities = paragraph["entities"]
                paragraph["entities"] = [entity for entity in entities if entity["word"] not in news_site_to_entities[website]]


    @staticmethod
    def remove_duplicate_entities_with_diff_groups(dataset : ParagraphDataset):
        """
        For each paragraph in article, search for entities with same 'word' but diff
        'entity_group'. Convert entity groups of duplicate 'word' ents to the one with the highest count.
        """
        
        # create a new dataset and convert to article format
        article_dataset = copy.deepcopy(dataset)
        DatasetUtils.to_article_dataset(article_dataset)

        # in article_dataset, each paragraph is an article
        new_data = []
        for paragraph in dataset.data:
            article = article_dataset.get_filtered_by_key_vals("article_id", paragraph["article_id"])[0]
            for entity in paragraph["entities"]:
                same_word_ents = [e for e in article["entities"] if e["word"].lower() == entity["word"].lower()]
                if len(same_word_ents) > 1:
                    # if multiple ents with same word, change entity group to the one with highest count
                    entity_group_counts = defaultdict(int)
                    for e in same_word_ents:
                        entity_group_counts[e["entity_group"]] += 1
                    entity["entity_group"] = max(entity_group_counts, key=entity_group_counts.get)
            new_data.append(paragraph)
        dataset.set_data(new_data)
    



