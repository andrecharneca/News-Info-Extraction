import pandas as pd
from pprint import pprint
import itertools
import sys 
sys.path.append("../../src/")

from pipelines.utils.relationships import (RELATIONSHIP_TO_SENTENCE_DICT,
                                            UNIDIR_RELATIONSHIP_LIST,
                                            BIDIR_RELATIONSHIP_LIST,
                                            UNIDIR_STRING_FORMAT,
                                            BIDIR_STRING_FORMAT,
                                            ORG_ORG_RELATIONSHIP_LIST,
                                            ORG_PRODUCT_RELATIONSHIP_LIST,
                                            PRODUCT_PRODUCT_RELATIONSHIP_LIST,
)
                                            
from pipelines.utils.entities import (ENTITY_TYPE_TO_WORD_DICT,
                                    ENTITY_FILTER_STRING_FORMAT,)
from pipelines.utils.newspaper_utils import get_article
                   
class ParagraphUtils(object):
    
    @staticmethod
    def text_to_paragraphs(text : str) -> list[str]:
        """
        Splits text into paragraphs
        args:
            text (str): text
        returns:
            list[str]: list of paragraphs
        """
        paragraphs = text.split("\n")
        return paragraphs

    @staticmethod
    def filter_paragraphs(paragraphs : list[str]) -> list[str]:
        """
        Filters the paragraphs of an article
        args:
            paragraphs (list[str]): list of paragraphs
        returns:
            list[str]: list of filtered paragraphs
        """
        # remove paragraphs with 3 words or less
        paragraphs = [p for p in paragraphs if len(p.split()) > 3]
        
        # remove duplicated paragraphs, but keep order
        paragraphs = list(dict.fromkeys(paragraphs))
        return paragraphs

    @staticmethod
    def concat_paragraphs(tokenizer, paragraphs : list[str], max_tokens: int = 256) -> list[str]:
        """
        Concatenates paragraphs until the max_tokens is reached
        args:
            tokenizer (transformers.PreTrainedTokenizer preferably): tokenizer
            paragraphs (list[str]): list of paragraphs
            max_tokens (int): max tokens for concated paragraphs
                            NOTE: if 1 paragraph is longer than max_tokens, it will be returned as is 
        returns:
            list[str]: list of paragraphs
        """
        paragraphs = [tokenizer.encode(p) for p in paragraphs]

        # concatenate paragraphs until max_tokens is reached
        # we dont keep the [CLS] and [SEP] tokens, bc the sentences will be tokenized again soon
        paragraphs_tokens = []
        paragraph_tokens = []
        for paragraph in paragraphs:
            if len(paragraph_tokens) + len(paragraph) < max_tokens -2: # -2 for [CLS] and [SEP]
                paragraph_tokens += paragraph[1:-1]
            else:
                paragraphs_tokens.append(paragraph_tokens)
                paragraph_tokens = paragraph[1:-1]
        paragraphs_tokens.append(paragraph_tokens)

        # decode paragraphs
        paragraphs = [tokenizer.decode(p) for p in paragraphs_tokens]

        return paragraphs

    @staticmethod
    def get_all_urls(df_path : str) -> list[str]:
        """
        Gets all the urls from the articles dataframe
        args:
            df_path (str): path to the articles dataframe
        returns:
            list[str]: list of urls
        """
        df_articles = pd.read_csv(df_path)
        urls = df_articles["url"].unique().tolist()
        return urls


class RelationshipUtils(object):
    @staticmethod
    def get_triples(entities : list[dict], 
                                unidir_relationship_list : list[str] = UNIDIR_RELATIONSHIP_LIST,
                                bidir_relationship_list : list[str] = BIDIR_RELATIONSHIP_LIST,
                                org_org_relationship_list : list[str] = ORG_ORG_RELATIONSHIP_LIST,
                                org_product_relationship_list : list[str] = ORG_PRODUCT_RELATIONSHIP_LIST,
                                product_product_relationship_list : list[str] = PRODUCT_PRODUCT_RELATIONSHIP_LIST,
                                ) -> list[dict]:
        """
        Gets all the possible relationship triples between the entities.
        This was designed with companies in mind.
        args:
            entities (list[dict]): list of entities, at least with key "word", "entity_group"
            unidir_relationship_list (list[str]): list of unidirectional relationships
            bidir_relationship_list (list[str]): list of bidirectional relationships
            org_org_relationship_list (list[str]): list of relationships between two organizations
            org_product_relationship_list (list[str]): list of relationships between an organization and a product
            product_product_relationship_list (list[str]): list of relationships between two products
        returns:
            list[dict]: list of relationship triples {entity_1, entity_2, relationship}
        """
        orgs = [entity["word"] for entity in entities if entity["entity_group"] == "ORG"]
        products = [entity["word"] for entity in entities if entity["entity_group"] == "PROD"]
        rel_lists = {"org-org": org_org_relationship_list,
                    "org-product": org_product_relationship_list,
                    "product-product": product_product_relationship_list}
        relationship_triples = []

        # bidir -> order doesnt matter
        bidir_entity_tuples = {"org-org": list(itertools.combinations(orgs, 2)),
                                "org-product": list(itertools.product(orgs, products)),
                                "product-product": list(itertools.combinations(products, 2))}
                               

        # unidir -> order matters
        unidir_entity_tuples = {"org-org": [(entity_1, entity_2) for entity_1 in orgs for entity_2 in orgs if entity_1 != entity_2],
                                "org-product": [(entity_1, entity_2) for entity_1 in orgs for entity_2 in products],
                                "product-product": [(entity_1, entity_2) for entity_1 in products for entity_2 in products if entity_1 != entity_2]}

        for key in unidir_entity_tuples.keys():
            relationship_triples += [{"entity_1": entity_1, "entity_2": entity_2, "relationship": relationship}
                                    for entity_1, entity_2 in unidir_entity_tuples[key]
                                    for relationship in set(rel_lists[key]) & set(unidir_relationship_list)]
        
        for key in bidir_entity_tuples.keys():
            relationship_triples += [{"entity_1": entity_1, "entity_2": entity_2, "relationship": relationship}
                                    for entity_1, entity_2 in bidir_entity_tuples[key]
                                    for relationship in set(rel_lists[key]) & set(bidir_relationship_list)]

        return relationship_triples
    
    @staticmethod
    def triples_to_sentences(triples : list[dict],
                                   unidir_relationship_list : list[str] = UNIDIR_RELATIONSHIP_LIST,
                                   bidir_relationship_list : list[str] = BIDIR_RELATIONSHIP_LIST,
                                   unidir_string_format : str = UNIDIR_STRING_FORMAT,
                                   bidir_string_format : str = BIDIR_STRING_FORMAT,
                                   relationship_to_sentence_dict : dict = RELATIONSHIP_TO_SENTENCE_DICT,
                                   ) -> list[str]:
        """
        Turns list of relationship triples into sentences.
        Orders the entities in the sentence alphabetically.
        """
        unidir_sentences = [unidir_string_format.format(entity_1 = triple["entity_1"],
                                                        relationship = relationship_to_sentence_dict[triple["relationship"]],
                                                        entity_2 = triple["entity_2"])
                            for triple in triples
                            if triple["relationship"] in unidir_relationship_list]
        
        bidir_sentences = [bidir_string_format.format(entity_1 = min(triple["entity_1"], triple["entity_2"]),
                                              relationship = relationship_to_sentence_dict[triple["relationship"]],
                                              entity_2 = max(triple["entity_1"], triple["entity_2"]))
                   for triple in triples
                   if triple["relationship"] in bidir_relationship_list]
        sentences = unidir_sentences + bidir_sentences
        return sentences
    
    @staticmethod
    def get_relationships(entities : list[dict], 
                                unidir_relationship_list : list[str] = UNIDIR_RELATIONSHIP_LIST,
                                bidir_relationship_list : list[str] = BIDIR_RELATIONSHIP_LIST,
                                unidir_string_format : str = UNIDIR_STRING_FORMAT,
                                bidir_string_format : str = BIDIR_STRING_FORMAT,
                                relationship_to_sentence_dict : dict = RELATIONSHIP_TO_SENTENCE_DICT,
                                ) -> list[dict]:
        """
        Gets all the possible relationships between the entities.
        This was designed with companies in mind.
        args:
            entities (list[dict]): list of entities
            unidir_relationship_list (list[str]): list of unidirectional relationships
            bidir_relationship_list (list[str]): list of bidirectional relationships
            unidir_string_format (str): string format for unidirectional relationships
            bidir_string_format (str): string format for bidirectional relationships
            relationship_to_sentence_dict (dict): dictionary mapping relationships to sentences
        returns:
            list[dict]: list of relationship triples {entity_1, entity_2, relationship, sentence}
        """
        triples = RelationshipUtils.get_triples(entities, 
                                                unidir_relationship_list = unidir_relationship_list,
                                                bidir_relationship_list = bidir_relationship_list,)
        sentences = RelationshipUtils.triples_to_sentences(triples,
                                                            unidir_relationship_list = unidir_relationship_list,
                                                            bidir_relationship_list = bidir_relationship_list,
                                                            unidir_string_format = unidir_string_format,
                                                            bidir_string_format = bidir_string_format,
                                                            relationship_to_sentence_dict = relationship_to_sentence_dict,)
        relationships = [{"entity_1": triple["entity_1"],
                            "entity_2": triple["entity_2"],
                            "relationship": triple["relationship"],
                            "sentence": sentence}
                            for triple, sentence in zip(triples, sentences)]
        
        return relationships
    
    @staticmethod
    def get_entity_filtering_relationships(entities : list[dict],
                                           entity_type_to_word_dict : dict = ENTITY_TYPE_TO_WORD_DICT,
                                           entity_filter_string_format : str = ENTITY_FILTER_STRING_FORMAT,
                                           ) -> list[dict]:
        """
        Receives list of entities and returns a list of relationships for filtering purposes.
        Outputs {"sentence": "[ORG] is a company [or product].", "entity_word": "[word]"} for each entity.
        """
        relationships = [{
            "sentence": entity_filter_string_format.format(entity = entity["word"], entity_type = entity_type_to_word_dict[entity["entity_group"]]),
            "entity_word": entity["word"],
        } for entity in entities]
        return relationships



