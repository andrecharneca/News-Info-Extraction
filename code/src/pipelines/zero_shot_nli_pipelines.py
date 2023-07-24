from typing import Union
from pprint import pprint
import numpy as np
from transformers import (AutoModelForTokenClassification,
                          AutoModelForSequenceClassification, 
                          AutoTokenizer,
                          TokenClassificationPipeline,
                          TextClassificationPipeline)
from .utils.entities import NATIONALITIES_LIST
import cleanco


class CompanyNERPipeline(TokenClassificationPipeline):
    """
    Extracts entities from paragraphs.
    Paragraph dict (or list[dict]) -> NER Extraction -> Post processing/filtering -> entities (list[dict])
        - entitities = list[dict {"word": str, "score": float, "entity_group": str}]
        - paragraph must have (at least) the following keys: text
    """
    def __init__(self, *args, 
                 threshold : float = 0.95, 
                 keep_groups : Union[str, list[str]] = ['ORG', 'MISC'],
                 exclude_from_list : list[str] = NATIONALITIES_LIST, 
                 remove_duplicates : bool = True, 
                 **kwargs):
        
        if isinstance(kwargs['model'], str):
            kwargs['model'] = AutoModelForTokenClassification.from_pretrained(kwargs['model'])
            kwargs['tokenizer'] = AutoTokenizer.from_pretrained(kwargs['model'].name_or_path)

        # ner params
        self.threshold = threshold
        self.keep_groups = keep_groups
        self.exclude_from_list = exclude_from_list
        self.remove_duplicates = remove_duplicates

        super().__init__(*args, **kwargs)
    
    def preprocess(self, inputs, **kwargs):
        """
        Receives paragraph = Dict{..., "text": str, ...}
        """
        text = inputs["text"]
        return super().preprocess(text, **kwargs)

    def postprocess(self, model_outputs, **kwargs):
        """
        - apply threshold
        - keep only entities in keep_groups
        - remove legal suffixes
        - remove duplicates
        """
        entities_dict = super().postprocess(model_outputs, **kwargs)  

        if isinstance(self.keep_groups, str):
            self.keep_groups = [self.keep_groups]

        entities_dict = [entity for entity in entities_dict 
                         if entity['entity_group'] in self.keep_groups
                         if entity['word'] not in self.exclude_from_list
                         if entity['score'] >= self.threshold]

        filtered_entities = []
        for entity in entities_dict:
            #remove start and end keys
            entity.pop('start')
            entity.pop('end')
            # convert to float to avoid json serialization error
            entity['score'] = float(entity['score'])
            # remove legal suffixes
            entity['word'] = cleanco.basename(entity['word']) 
            
            # replace MISC with PROD
            if entity['entity_group'] == 'MISC':
                entity['entity_group'] = 'PROD'

            # remove words with length <= 2
            if len(entity['word']) > 2:
                filtered_entities.append(entity)

        if self.remove_duplicates:
            # remove duplicates, keep highest score
            filtered_entities = sorted(filtered_entities, key=lambda x: x["score"], reverse=True)
            filtered_entities = {entity["word"]: entity for entity in filtered_entities}
            filtered_entities = list(filtered_entities.values())

        # output a list of dicts with keys: word, score, entity_group
        return filtered_entities
    
    def __call__(self, paragraph, *args, **kwargs):
        return super().__call__(paragraph, *args, **kwargs)

class CompanyZeroShotClassificationPipeline(TextClassificationPipeline):
    """
    Ranks possible relationships in a paragraph.
    Paragraph (dict, list[dict], ParagraphDataset) -> NLI classification -> Post processing/filtering -> relationships (list[dict])
        - relationships = list[dict{"entity_1"(opt): str, "entity_2"(opt): str, "relationship"(opt): str, "sentence": str, "score": float}]
        - paragraph must have (at least) the following keys: text, relationships (only 1 relationship)
    """
    def __init__(self, *args,
                    hypothesis_template : str = "According to this example: {}",
                    **kwargs):
        
        kwargs = self._sanitize_kwargs(**kwargs)

        # zero shot params
        self.hypothesis_template = hypothesis_template
        super().__init__(*args, **kwargs)

    def _sanitize_kwargs(self, **kwargs):
        if isinstance(kwargs['model'], str):
            kwargs['model'] = AutoModelForSequenceClassification.from_pretrained(kwargs['model'])
            kwargs['tokenizer'] = AutoTokenizer.from_pretrained(kwargs['model'].name_or_path)
        return kwargs
    
    def preprocess(self, paragraph, **kwargs):
        """
        Receives paragraph with only 1 relationship.
        """
        inputs = {
            "text": paragraph["text"],
            "text_pair": self.hypothesis_template.format(paragraph["relationships"][0]["sentence"]),
        }
        return super().preprocess(inputs, **kwargs)
    
    def postprocess(self, model_outputs, *args, **kwargs):
        """
        - convert from entailed/contradiction logits to score
        """
        logits = model_outputs['logits'][0]
        entailment_id = self.model.config.label2id["entailment"]
        contradiction_id = self.model.config.label2id["contradiction"]
        entailment_score = logits[entailment_id]
        contradiction_score = logits[contradiction_id]
        score = np.exp(entailment_score) / (np.exp(entailment_score) + np.exp(contradiction_score))

        return float(score.item())
    
    def __call__(self, paragraph, *args, **kwargs):
        """
        Returns scores for each paragraphs relationship, in list
        Notes:
         - make sure to run dataset.expand_relationships() before calling this function
         - paragraphs received by this function should have only 1 relationship each (for batching purposes)
        """
        return super().__call__(paragraph, top_k=None, *args, **kwargs)

    