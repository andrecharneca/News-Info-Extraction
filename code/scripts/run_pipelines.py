import sys
from tqdm import tqdm
from time import time
import os
from pprint import pprint

# see the amount of memry used by the GPU
import GPUtil

sys.path.append("../src/")
from pipelines.paths import (ARTICLES_JSONS_PATH,
                              RESULTS_DATA_DIR,
                              )
from pipelines.zero_shot_nli_pipelines import CompanyNERPipeline, CompanyZeroShotClassificationPipeline
from pipelines.gpt3_pipeline import CompanyGPT3Pipeline
from pipelines.datamodules.datasets import ParagraphDataset
from pipelines.utils.prompts import ENTITY_REL_ENTITY_WITH_ENTITY_LIST_PROMPT
from pipelines.utils.utils import ZeroShotUtils, DatasetUtils, EntityUtils

from torch.utils.data import DataLoader

## ZSC Params ##
zero_shot_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ner_model_name = "xlm-roberta-large-finetuned-conll03-english"
max_tokens = 128
ner_batch_size = 16
zsc_batch_size = 16
zsc_threshold = 0.995
ner_threshold = 0.995
zsc_entity_filter_threshold = 0.99
device = 9

## GPT3 Params ##
prompt = ENTITY_REL_ENTITY_WITH_ENTITY_LIST_PROMPT
debug = True
gpt_kwargs = {
    "model": "text-davinci-003",
    "temperature": 0,
    "max_tokens": 512,
    "top_p": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

def run_gpt3_pipeline():
    dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
                               article_id_list = None,
                                 tokenizer_model_name = zero_shot_model_name,
                                    max_tokens = 1e6)
    start = time()
    gpt3_pipe = CompanyGPT3Pipeline(prompt = prompt, 
                                    debug=debug, 
                                    **gpt_kwargs)
    output = gpt3_pipe(dataset)

    dataset.add_key_vals([out["relationships"] for out in output], key="relationships")
    dataset.add_key_vals([out['entities'] for out in output], key="entities")
    DatasetUtils.order_bidir_relationships_entities(dataset)

    # save to results
    DatasetUtils.to_article_dataset(dataset)
    folder = os.path.join(RESULTS_DATA_DIR, "gpt3")
    dataset.to_jsons(folder)
    end = time()
    print(f"GPT Time: {(end-start):.2f}s")

def run_zsc_pipeline(zero_shot_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                        ner_model_name = "xlm-roberta-large-finetuned-conll03-english",
                        max_tokens = 128,
                        ner_batch_size = 16,
                        zsc_batch_size = 16,
                        zsc_threshold = 0.99,
                        ner_threshold = 0.99,
                        zsc_entity_filter_threshold = 0.99,
                        device = 9, 
                        folder_name = "zsc",
                        verbose = True):
    dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
                                tokenizer_model_name = zero_shot_model_name,
                                article_id_list = None,
                                 max_tokens = max_tokens)
    
    if verbose: print(f"Number of articles: {len(dataset.get_key_vals('article_id'))}")

    ## Stage 1 - NER ##
    ner_pipe = CompanyNERPipeline(model = ner_model_name,
                                    device = device,
                                    aggregation_strategy = "simple",
                                    threshold = ner_threshold,)
    if verbose: print("Getting entities...")
    ner_start = time()
    entities = []
    for output in tqdm(ner_pipe(dataset, batch_size=ner_batch_size), total=len(dataset)):
        entities.append(output)
    ner_end = time()
    del ner_pipe # free ner_pipe gpu memory

    dataset.add_key_vals(entities, key="entities")
    EntityUtils.remove_news_site_entities(dataset)

    ## Stage 2 - Entity Linking ##
    if verbose: print(f"\nEntities/paragraph before linking: {sum([len(paragraph['entities']) for paragraph in dataset])/len(dataset):.1f}")
    if verbose: print("Linking entities...")
    EntityUtils.apply_entity_linking(dataset)
    if verbose: print(f"Entities/paragraph after linking: {sum([len(paragraph['entities']) for paragraph in dataset])/len(dataset):.1f}")
    
    ## Stage 3 - Entity Filtering ##
    zero_shot_pipe = CompanyZeroShotClassificationPipeline(model = zero_shot_model_name, device = device)
    if verbose: print("\nFiltering entities...")
    ZeroShotUtils.compute_entity_filtering_relationships_and_add(dataset)
    DatasetUtils.expand_relationships(dataset)
    DatasetUtils.remove_paragraphs_without_relationships(dataset) # this effectively removes paragraphs without entities

    dataloader = DataLoader(dataset,
                            batch_size=zsc_batch_size,
                            shuffle=False,
                            collate_fn=dataset.zsc_collate_fn,)

    start = time()
    entity_filter_relationships_scores = []
    for batch in tqdm(dataloader):
        entity_filter_relationships_scores += zero_shot_pipe(batch, batch_size=zsc_batch_size)
    end = time()

    ZeroShotUtils.add_scores(dataset, entity_filter_relationships_scores)
    DatasetUtils.contract_relationships(dataset)
    ZeroShotUtils.filter_entities_by_relationship_score(dataset, zsc_entity_filter_threshold)
    dataset.remove_key_from_data("relationships")
    EntityUtils.remove_duplicate_entities_with_diff_groups(dataset)   
    if verbose: print(f"\nEntities/paragraph after filtering: {sum([len(paragraph['entities']) for paragraph in dataset])/len(dataset):.1f}")

    ## Stage 4 - Relationship Extraction ##
    ZeroShotUtils.compute_relationships_and_add(dataset)
    DatasetUtils.expand_relationships(dataset)
    DatasetUtils.remove_paragraphs_without_relationships(dataset)

    if verbose: print(f"\nTotal number of relationships to score: {sum([len(paragraph['relationships']) for paragraph in dataset.data])}")
    
    dataloader = DataLoader(dataset,
                            batch_size=zsc_batch_size,
                            shuffle=False,
                            collate_fn=dataset.zsc_collate_fn,)
    
    if verbose: print("\nGetting relationship scores...")
    
    start = time()
    output = []
    for batch in tqdm(dataloader):
        output += zero_shot_pipe(batch, batch_size=zsc_batch_size)
    end = time()

    ZeroShotUtils.add_scores(dataset, output)
    DatasetUtils.contract_relationships(dataset)
    ZeroShotUtils.filter_relationships_by_score(dataset, zsc_threshold)
    DatasetUtils.to_article_dataset(dataset)
    ZeroShotUtils.keep_highest_score_duplicate_rels(dataset)
    DatasetUtils.order_bidir_relationships_entities(dataset)

    if verbose: print(f"Total number of relationships: {sum([len(paragraph['relationships']) for paragraph in dataset.data])}")    
    if verbose: print(f"NER Time: {ner_end-ner_start:.2f}s")
    if verbose: print(f"ZSC Time: {end-start:.2f}s")
    if verbose: print(f"ZSC Memory usage: {GPUtil.getGPUs()[device].memoryUsed}MB")

    # to jsons
    folder = os.path.join(RESULTS_DATA_DIR, folder_name)
    dataset.to_jsons(folder)

def main():
    print("Running ZSC pipeline...\n")
    # run_zsc_pipeline()
    print("\nRunning GPT3 pipeline...\n")
    run_gpt3_pipeline()
    print("\nDone!")

if __name__ == "__main__":
    main()
