import sys
from tqdm import tqdm
from time import time
import os
from pprint import pprint
import copy
import cleanco
import parse
import json
# see the amount of memry used by the GPU
import GPUtil
import re
from difflib import SequenceMatcher


sys.path.append("../src/")
from pipelines.paths import (ARTICLES_JSONS_PATH,
                              TEMP_DATA_DIR,
                              RESULTS_DATA_DIR,
                              MANUAL_ANNOTATIONS_JSONS_PATH)
from pipelines.zero_shot_nli_pipelines import CompanyNERPipeline, CompanyZeroShotClassificationPipeline
from pipelines.gpt3_pipeline import CompanyGPT3Pipeline
from pipelines.datamodules.datasets import ParagraphDataset
from pipelines.datamodules.utils import RelationshipUtils
from pipelines.utils.prompts import ENTITY_REL_ENTITY_WITH_ENTITY_LIST_PROMPT
from pipelines.utils.entities import ENTITY_STOPWORDS
from pipelines.utils.utils import ZeroShotUtils, DatasetUtils, GPT3Utils, MetricsUtils, EntityUtils

from torch.utils.data import Dataset, DataLoader
from transformers import pipeline, TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

zero_shot_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ner_model_name = "xlm-roberta-large-finetuned-conll03-english"
max_tokens = 128
ner_batch_size = 16
zsc_batch_size = 16
zsc_threshold = 0.99
ner_threshold = 0.99
zsc_entity_filter_threshold = 0.99
device = 9

def test_companynerpipeline(ner_batch_size = ner_batch_size, max_tokens = max_tokens):
    dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
                               tokenizer_model_name = zero_shot_model_name,
                               max_tokens = max_tokens)
    ner_pipe = CompanyNERPipeline(model = ner_model_name,
                                  device = device,
                                  aggregation_strategy = "simple",)

    print(f"Number of paragraphs: {len(dataset)}")
    start = time()
    outputs = []
    for output in tqdm(ner_pipe(dataset, batch_size=ner_batch_size), total=len(dataset)):
        outputs.append(output)
    end = time()
    dataset.add_entities(outputs)
    print(f"Dataset[0]: {dataset[0]}")
    print(f"Outputs size: {len(outputs)}")
    print(f"Outputs[0]: {outputs[0]}")
    # #show 5 random paragraphs
    # ex_paragraphs = np.random.choice(dataset, size=5, replace=False)
    # for paragraph in ex_paragraphs:
    #     print(f"\nParagraph: {paragraph['text']}")
    #     print(f"\tEntities: {paragraph['entities']}")

    # # show 5 last paragraphs
    # for paragraph in dataset[-5:]:
    #     print(f"URL: {paragraph['url']}")
    #     print(f"\nParagraph: {paragraph['text']}")
    #     print(f"\tEntities: {paragraph['entities']}")

    print("\n\n")
    print(f"NER Time: {end-start:.2f}s")
    print(f'NER Memory usage: {GPUtil.getGPUs()[device].memoryUsed}MB')

def test_paragraphdataset():
    # dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
    #                            tokenizer_model_name = zero_shot_model_name,
    #                            max_tokens = max_tokens)
    # #dataset.expand_relationships()
    # print(f"Number of paragraphs: {len(dataset)}")
    # print(f"Keys: {dataset.keys}")
    # print(f"Dataset[0]: {dataset[0]}")
    dataset = DatasetUtils.load_csv('../../data/manual_annotations/manual_annotations.csv',
                                    relationship_keys = ["entity_1", "entity_2", "relationship", "passage"])
    print(f"Data[0]")
    pprint(dataset[0])

def test_relationshiputils():
    entities = [{'word':"Apple", 'entity_group':"ORG"} , {'word':"Google", 'entity_group':"ORG"},
                {'word':"iPhone", 'entity_group':"MISC"}, {'word':"Android", 'entity_group':"MISC"}]
    triples = RelationshipUtils.get_triples(entities)
    print(f"Triples: {triples}")
    sentences = RelationshipUtils.triples_to_sentences(triples)
    print(f"Sentences: {sentences}")
    relationships = RelationshipUtils.get_relationships(entities)
    print(f"Relationships: {relationships}")

    filtering_relationships = RelationshipUtils.get_entity_filtering_relationships(entities)
    print(f"Filtering relationships: {filtering_relationships}")
    
def test_companyzeroshotclassificationpipeline(max_tokens = max_tokens, ner_batch_size = ner_batch_size, zsc_batch_size = zsc_batch_size):
    dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
                                tokenizer_model_name = zero_shot_model_name,
                                article_id_list = [5],
                                 max_tokens = max_tokens)

    ## Stage 1 - NER ##
    ner_pipe = CompanyNERPipeline(model = ner_model_name,
                                    device = device,
                                    aggregation_strategy = "simple",
                                    threshold = ner_threshold,)
    print("Getting entities...")
    ner_start = time()
    entities = []
    for output in tqdm(ner_pipe(dataset, batch_size=ner_batch_size), total=len(dataset)):
        entities.append(output)
    ner_end = time()
    del ner_pipe # free ner_pipe gpu memory

    dataset.add_key_vals(entities, key="entities")
    EntityUtils.remove_news_site_entities(dataset)

    ## Stage 2 - Entity Linking ##
    print(f"\nEntities/paragraph before linking: {sum([len(paragraph['entities']) for paragraph in dataset])/len(dataset):.1f}")
    print("Linking entities...")
    EntityUtils.apply_entity_linking(dataset)
    print(f"Entities/paragraph after linking: {sum([len(paragraph['entities']) for paragraph in dataset])/len(dataset):.1f}")
    
    ## Stage 3 - Entity Filtering ##
    zero_shot_pipe = CompanyZeroShotClassificationPipeline(model = zero_shot_model_name, device = device)
    print("\nFiltering entities...")
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
    print(f"\nEntities/paragraph after filtering: {sum([len(paragraph['entities']) for paragraph in dataset])/len(dataset):.1f}")

    ## Stage 4 - Relationship Extraction ##
    ZeroShotUtils.compute_relationships_and_add(dataset)
    DatasetUtils.expand_relationships(dataset)
    DatasetUtils.remove_paragraphs_without_relationships(dataset)

    print(f"\nTotal number of relationships to score: {sum([len(paragraph['relationships']) for paragraph in dataset.data])}")
    
    dataloader = DataLoader(dataset,
                            batch_size=zsc_batch_size,
                            shuffle=False,
                            collate_fn=dataset.zsc_collate_fn,)
    
    print("\nGetting relationship scores...")
    
    start = time()
    output = []
    for batch in tqdm(dataloader):
        output += zero_shot_pipe(batch, batch_size=zsc_batch_size)
    end = time()

    ZeroShotUtils.add_scores(dataset, output)
    DatasetUtils.contract_relationships(dataset)
    ZeroShotUtils.filter_relationships_by_score(dataset, zsc_threshold)
    DatasetUtils.to_article_dataset(dataset)
    pprint(dataset.data)
    ZeroShotUtils.keep_highest_score_duplicate_rels(dataset)
    DatasetUtils.order_bidir_relationships_entities(dataset)
    pprint(dataset.data)

    print(f"\nAfter filtering: {len(dataset)} paragraphs")
    print(f"Total number of relationships: {sum([len(paragraph['relationships']) for paragraph in dataset.data])}")    
    print(f"ZSC Time: {end-start:.2f}s")
    print(f"ZSC Memory usage: {GPUtil.getGPUs()[device].memoryUsed}MB")

    # to jsons
    # folder = os.path.join(RESULTS_DATA_DIR, "zsc_pipeline")
    # dataset.to_jsons(folder)

def test_zscbatching():
    zero_shot_pipe = CompanyZeroShotClassificationPipeline(model = zero_shot_model_name,
                                                            device = device,
                                                            threshold = 0)
    data = [{"url": "site1.com", "text": "Site1 par1", "relationships": [{"sentence": "Site1 par1 rel1."}]},
            {"url": "site1.com", "text": "Site1 par1", "relationships": [{"sentence": "Site1 par1 rel2."}]},
            {"url": "site1.com", "text": "Site1 par2", "relationships": [{"sentence": "Site1 par2 rel1."}]},
            {"url": "site2.com", "text": "Site2 par1", "relationships": [{"sentence": "Site2 par1 rel1."}]},
            {"url": "site2.com", "text": "Site2 par2", "relationships": [{"sentence": "Site2 par2 rel1."}]},
            {"url": "site2.com", "text": "Site2 par2", "relationships": [{"sentence": "Site2 par2 rel2."}]},
            {"url": "site2.com", "text": "Site2 par2", "relationships": []},
    ]

    dataset = ParagraphDataset(data = data,
                                 tokenizer_model_name = zero_shot_model_name,
                                 max_tokens = max_tokens)
    dataset.remove_paragraphs_without_relationships()
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            collate_fn=dataset.zsc_collate_fn,)
    
    # for i, batch in enumerate(dataloader):
    #     print(f"\nBatch {i}")
    #     for item in zero_shot_pipe.preprocess(batch):
    #         print("Sequence: ", item["sequence"])
    #         print("Candidate label: ", item["candidate_label"])
    for output in zero_shot_pipe(dataset, batch_size=4):
        print(output)

def test_textclassification(textclass_batch_size = 4, num_workers=0):
    # inputs = [{"text": "I love football.", "text_pair": "According to this example: he likes football"}, 
    #           {"text": "I love football.", "text_pair": "According to this example: he likes basketball"},
    #           {"text": "I love football.", "text_pair": "According to this example: he likes tennis"},
    #           {"text": "I love football.", "text_pair": "According to this example: he likes baseball"},
    #           {"text": "Carl's cat is white", "text_pair": "According to this example: he has animals"},
    #           {"text": "Carl's cat is white", "text_pair": "According to this example: he has a dog"},
    #           {"text": "Carl's cat is white", "text_pair": "According to this example: he likes the color black"},]
    dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
                                tokenizer_model_name = zero_shot_model_name,
                                max_tokens = max_tokens)

    # run ner
    ner_pipe = CompanyNERPipeline(model = ner_model_name,
                                    device = device,
                                    aggregation_strategy = "simple",)
    entities = []
    for output in tqdm(ner_pipe(dataset, batch_size=ner_batch_size), total=len(dataset)):
        entities.append(output)

    # free ner_pipe gpu memory
    del ner_pipe

    # expand relationships
    dataset.add_key_vals(entities, key="entities")
    dataset.compute_relationships_and_add()
    dataset.expand_relationships()
    dataset.remove_paragraphs_without_relationships()

    # get inputs = List[dict{"text": str, "text_pair": str}] where text_pair is the relationship sentence
    inputs = []
    hypothesis_template = "According to this example: {}"
    for paragraph in dataset.data:
        for rel in paragraph["relationships"]:
            inputs.append({"text": paragraph["text"], "text_pair": hypothesis_template.format(rel["sentence"])})
    
    print("Inputs[0]: ", inputs[0])
    print("Number of sentence pairs: ", len(inputs))

    # pipe = pipeline("text-classification", model=zero_shot_model_name, device=device, top_k=None)
    
    class MyTextClassificationPipeline(TextClassificationPipeline):
        def __init__(self, *args, **kwargs):
            kwargs["model"] = AutoModelForSequenceClassification.from_pretrained(kwargs["model"])
            kwargs['tokenizer'] = AutoTokenizer.from_pretrained(kwargs['model'].name_or_path)
            super().__init__(*args, **kwargs)
        def postprocess(self, model_outputs, **kwargs):
            print(model_outputs)
            return super().postprocess(model_outputs, **kwargs)
    
    pipe = MyTextClassificationPipeline(model=zero_shot_model_name, device=device, top_k=None)

    # measure time and memory
    start = time()
    outputs = []
    class InputsDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    inputs_dataset = InputsDataset(inputs)
    for output in tqdm(pipe(inputs_dataset, batch_size=textclass_batch_size), total=len(inputs_dataset)):
        outputs.append(output)
    end = time()
    print("Outputs[0]:", outputs[0])
    print(f"Number of outputs: {len(outputs)}")
    print(f"Time: {end-start:.2f}s")
    print(f"Memory usage: {GPUtil.getGPUs()[device].memoryUsed}MB")

def test_gpt3pipeline():
    dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
                               article_id_list = None,
                                 tokenizer_model_name = zero_shot_model_name,
                                    max_tokens = 1e6)
    
    print(f"Number of paragraphs: {len(dataset)}")

    gpt_kwargs = {
        "model": "text-davinci-003",
        "temperature": 0,
        "max_tokens": 400,
        "top_p": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    }
    gpt3_pipe = CompanyGPT3Pipeline(prompt = ENTITY_REL_ENTITY_WITH_ENTITY_LIST_PROMPT, 
                                    debug=False, 
                                    **gpt_kwargs)
    output = gpt3_pipe(dataset)

    dataset.add_key_vals([out["relationships"] for out in output], key="relationships")
    dataset.add_key_vals([out['entities'] for out in output], key="entities")

    # print all relationships
    for paragraph in dataset.data:
        print(f"\n\n article_id: {paragraph['article_id']}")
        print(f"Relationships:")
        pprint(paragraph["relationships"])
        print(f"Entities:")
        pprint(paragraph["entities"])
    
    # save to results
    # DatasetUtils.to_article_dataset(dataset)
    # folder = os.path.join(RESULTS_DATA_DIR, "gpt3_with_ent_list")
    # dataset.to_jsons(folder)

def test_metrics():
    # pred_data = [{"article_id":2, 
    #                 "paragraph_id":1,
    #               "relationships": [
    #                   {"entity_1": "Microsoft", "entity_2": "Apple", "relationship": "competitor"},
    #                   {"entity_1": "Google", "entity_2": "Android", "relationship": "developer of"},
    #                   {"entity_1": "Google", "entity_2": "Apple", "relationship": "partner of"},
    #                   {"entity_1": "Microsoft", "entity_2": "OpenAI", "relationship": "owner of"},
    #                   {"entity_1": "Facebook", "entity_2": "Google", "relationship": "rebranded as"},
    #                   ]},
    #                 {"article_id":3,
    #                 "paragraph_id":2,
    #                 "relationships": [
    #                     {"entity_1": "Google", "entity_2": "Android", "relationship": "developer of"},
    #                     ]},
    #                 ]
    # target_data = [{"article_id":2,
    #                 "paragraph_id":0,
    #                 "relationships": [
    #                     {"entity_1": "Microsoft", "entity_2": "Apple", "relationship": "competitor"},
    #                     {"entity_1": "Google", "entity_2": "Android", "relationship": "developer of"},
    #                     {"entity_1": "Google", "entity_2": "Apple", "relationship": "competitor"},
    #                     {"entity_1": "Microsoft", "entity_2": "OpenAI", "relationship": "investor in"},
    #                     {"entity_1": "Facebook", "entity_2": "Meta", "relationship": "rebranded as"},
    #                     {"entity_1": "Facebook", "entity_2": "Google", "relationship": "competitor"},
    #                     {"entity_1": "Microsoft", "entity_2": "Xbox", "relationship": "developer of"},
    #                     ]},
    #                 {"article_id":3,
    #                 "paragraph_id":0,
    #                 "relationships": [
    #                     {"entity_1": "Google", "entity_2": "Apple", "relationship": "partner of"},
    #                     ]},
    #                 ]
    # expected precision: 2/6 = 0.3333
    # expected recall: 2/8 = 0.25
    
    # ask for user input for article_id
    article_id_list = None
    zsc_predictions = DatasetUtils.load_jsons(folder = os.path.join(RESULTS_DATA_DIR, "zsc"),
                                           data_path = ARTICLES_JSONS_PATH,
                                            article_id_list = article_id_list,
                                            tokenizer_model_name = zero_shot_model_name,
                                            max_tokens = 1e6)
    gpt_predictions = DatasetUtils.load_jsons(folder = os.path.join(RESULTS_DATA_DIR, "gpt3"),
                                             data_path = ARTICLES_JSONS_PATH,
                                            article_id_list = article_id_list,
                                            tokenizer_model_name = zero_shot_model_name,
                                            max_tokens = 1e6)
    target = DatasetUtils.load_jsons(folder = MANUAL_ANNOTATIONS_JSONS_PATH,
                                            data_path = ARTICLES_JSONS_PATH,
                                            article_id_list = article_id_list,
                                            tokenizer_model_name = zero_shot_model_name,
                                            max_tokens = 1e6)
    DatasetUtils.order_bidir_relationships_entities(target)
    DatasetUtils.order_bidir_relationships_entities(gpt_predictions)
    EntityUtils.link_entities_to_target(gpt_predictions, target)
    
    article_ids = target.get_key_vals("article_id")
    for id in article_ids:
        print(f"\n\nArticle id: {id}, URL: {target.get_filtered_by_key_vals('article_id', id)[0]['url']}")
        target_article = ParagraphDataset(data=target.get_filtered_by_key_vals("article_id", id))
        gpt_article = ParagraphDataset(data=gpt_predictions.get_filtered_by_key_vals("article_id", id))
        zsc_article = ParagraphDataset(data=zsc_predictions.get_filtered_by_key_vals("article_id", id))

        print("\n---Target---\n")
        for e in target_article[0]["entities"]:
            print(f"{e['word']}", end=", ")
        print()
        for relationship in target_article[0]["relationships"]:
            print(f"{relationship['entity_1']},{relationship['relationship']},{relationship['entity_2']}")

        print("\n---ZSC---\n")
        if len(zsc_article) > 0:
            for e in zsc_article[0]["entities"]:
                print(f"{e['word']} ({e['entity_group']})", end=", ")
            print()
            for relationship in zsc_article[0]["relationships"]:
                print(f"{relationship['entity_1']},{relationship['relationship']},{relationship['entity_2']}, {relationship['score']}")
        metrics = MetricsUtils.compute_avg_article_metrics(zsc_article, target_article)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Intersection: {MetricsUtils.compute_intersection_relationships(zsc_article, target_article)}")

        print("\n---GPT3---\n")
        if len(gpt_article) > 0:
            for e in gpt_article[0]["entities"]:
                print(f"{e['word']} ({e['entity_group']})", end=", ")
            print()
            for relationship in gpt_article[0]["relationships"]:
                print(f"{relationship['entity_1']},{relationship['relationship']},{relationship['entity_2']}, {relationship['passage']}")
        metrics = MetricsUtils.compute_avg_article_metrics(gpt_article, target_article)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Intersection: {MetricsUtils.compute_intersection_relationships(gpt_article, target_article)}")

    print("\n---Total---\n")    
    total_metrics_zsc = MetricsUtils.compute_avg_article_metrics(zsc_predictions, target)
    total_metrics_gpt = MetricsUtils.compute_avg_article_metrics(gpt_predictions, target)
    print(f"ZSC Precision: {total_metrics_zsc['precision']:.4f}")
    print(f"ZSC Recall: {total_metrics_zsc['recall']:.4f}")
    print(f"ZSC F1: {total_metrics_zsc['f1']:.4f}")
    print(f"GPT3 Precision: {total_metrics_gpt['precision']:.4f}")
    print(f"GPT3 Recall: {total_metrics_gpt['recall']:.4f}")
    print(f"GPT3 F1: {total_metrics_gpt['f1']:.4f}")

def test_datasetutils():
    dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
                               article_id_list = None,)
    pprint(dataset.data)

def test_zeroshotutils():
    # create a small dataset
    data = [
        {
            "text": "Google and Apple are competitors. Google and Microsoft are partners.",
            "entities": [
                {"entity": "Google", "score": 0.999, "entity_group": "ORG"},
                {"entity": "Apple", "score": 0.999, "entity_group": "ORG"},
            ],
            "relationships": [
                {"entity_1": "Google", "entity_2": "Apple", "score": 0.9, "relationship": "competitor"},
                {"entity_1": "Google", "entity_2": "Apple", "score": 0.8, "relationship": "partner of"}
            ]
        },
        {
            "text": "Amazon and Facebook are competitors. Amazon and Microsoft are partners.",
            "entities": [
                {"entity": "Amazon", "score": 0.999, "entity_group": "ORG"},
                {"entity": "Facebook", "score": 0.999, "entity_group": "ORG"},
            ],
            "relationships": [
                {"entity_1": "Google", "entity_2": "Apple", "score": 0.99, "relationship": "investor in"},
                {"entity_1": "Amazon", "entity_2": "Facebook", "score": 0.7, "relationship": "investor in"},
                {"entity_1": "Amazon", "entity_2": "Facebook", "score": 0.95, "relationship": "partner of"}
            ]
        }
    ]

    # create a ParagraphDataset object from the data
    dataset = ParagraphDataset(data = data)

    # keep only the highest scoring relationship for each entity pair
    ZeroShotUtils.keep_highest_score_duplicates(dataset)

    # print the updated dataset
    print(dataset.data)

def test_entityfiltering():
    dataset = ParagraphDataset(data_path = ARTICLES_JSONS_PATH,
                               article_id_list = [3],
                                tokenizer_model_name = zero_shot_model_name,
                                max_tokens = max_tokens)
    
    ## Stage 1 - NER ##
    ner_pipe = CompanyNERPipeline(model = ner_model_name,
                                    device = device,
                                    aggregation_strategy = "simple",
                                    threshold = ner_threshold,)
    print("Getting entities...")
    ner_start = time()
    entities = []
    for output in tqdm(ner_pipe(dataset, batch_size=ner_batch_size), total=len(dataset)):
        entities.append(output)
    ner_end = time()
    del ner_pipe # free ner_pipe gpu memory

    dataset.add_key_vals(entities, key="entities")
    pre_filtering_dataset = copy.deepcopy(dataset)

    ## Stage 2 - Entity Filtering ##
    zero_shot_pipe = CompanyZeroShotClassificationPipeline(model = zero_shot_model_name,
                                                            device = device)
    print("Filtering entities...")
    ZeroShotUtils.compute_entity_filtering_relationships_and_add(dataset)
    DatasetUtils.expand_relationships(dataset)

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
    ZeroShotUtils.filter_entities_by_relationship_score(dataset, 0.99)
    
    for i,paragraph in enumerate(dataset.data):
        print("\nText:\n", paragraph["text"])
        print("\nPre filtering entities:\n")
        pprint(pre_filtering_dataset.get_paragraph(article_id=paragraph["article_id"], paragraph_id=paragraph["paragraph_id"])["entities"])
        print("\nPost filtering entities:\n")
        pprint(paragraph["entities"])
    
def test_entitylinking():
    data = [
        {
            "article_id": 0,
            "paragraph_id": 0,
            "entities": [
                {"word": "aeris", "entity_group": "ORG"},
                {"word": "Aeris", "entity_group": "ORG"},
                {"word": "Aeris Communications", "entity_group": "ORG"},
            ]
        },
        {
            "article_id": 0,
            "paragraph_id": 1,
            "entities": [
                {"word": "aeris communications", "entity_group": "ORG"},
            ]
        },
        {
            "article_id": 1,
            "paragraph_id": 0,
            "entities": [
                {"word": "the chatgpt", "entity_group": "MISC"},
                {"word": "ChatGPT", "entity_group": "MISC"},
            ]
        },
        {
            "article_id": 1,
            "paragraph_id": 1,
            "entities": [
                {"word": "chatgpt bot", "entity_group": "MISC"},
                {"word": "ChatGPT, the", "entity_group": "MISC"},
            ]
        },
    ]

    dataset = ParagraphDataset(data = data)
    EntityUtils.apply_entity_linking(dataset)
    # pprint(dataset.data)

def test_cleanco():
    entities = ["Ford Spain", "Aeris Group Services", "Amazon",
                "Aeris, Ltd", "Ford", "aeris group", "Aeris Communications", "Amazon Health Services"]
    entities_dicts = [{"word": entity, "entity_group": "ORG", "score":0.9} for entity in entities]

    for i in range(len(entities_dicts)):
        if i < len(entities_dicts)//2:
            entities_dicts[i]["paragraph_id"] = 0

        else:
            entities_dicts[i]["paragraph_id"] = 1
        
        if i%2 == 0:
            entities_dicts[i]["entity_group"] = "MISC"

    pprint(entities_dicts)

    # the goal is to condense this list into ["aeris", "ford"]
    def clean(company_name):
        result = company_name.lower()

        result = re.sub(r'[^\w\s]','',result) # remove punctuation
        result = cleanco.basename(result) # remove legal suffixes # DONT NEED THIS IN IMPLEMENTATION

        # split on whitespace and remove empty strings
        result = [word for word in result.split(" ") if word != ""]

        # remove stopwords
        result = [word for word in result if word not in ENTITY_STOPWORDS]

        return result
    
    def longest_match(a, b):
        return SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    # sort entities_dicts by length of word, from longest to shortest
    entities_dicts = sorted(entities_dicts, key=lambda x: len(x["word"]), reverse=True)

    for entity in entities_dicts:
        same_group_entities = [e for e in entities_dicts if e["entity_group"] == entity["entity_group"]
                               if e["word"] != entity["word"]]
        word_matches = []
        for e in same_group_entities:
            longest = longest_match(clean(entity["word"]), clean(e["word"]))
            if longest.size > 0 and longest.a == 0:
                word_matches.append(e["word"])

        # keep shortest match
        entity["word"] = min(word_matches, key=len) if len(word_matches) > 0 else entity["word"]        

    print()
    pprint(entities_dicts)


        
    
    # for i in range(len(clean_entities)):
    #     for j in range(i+1, len(clean_entities)):
    #         match = longest_match(clean_entities[i], clean_entities[j])
    #         # we only care if the match starts on the first word
    #         if match.size > 0 and match.a == 0:
    #             # print the match
    #             print(clean_entities[i], clean_entities[j], "Match:", clean_entities[i][match.a: match.a + match.size])

def test_parse():
    # article_id = 2
    # response = json.load(open(os.path.join(TEMP_DATA_DIR, "gpt3_responses", "article_id_{}_paragraph_id_0.json".format(article_id))))
    # text = response['choices'][0]['text']
    text = """[Amazon (inc) (ORG), One Medical (ORG), Federal Trade Commission (ORG), JPMorgan Chase (ORG), Berkshire Hathaway (ORG), PillPack (PROD), Haven (PROD)]
|Entity|Relationship|Entity|Passage|
|Amazon|owner of|One Medical|"Amazon closed its acquisition of health care provider One Medical and its parent in a $3.9 billion deal"|
|Amazon|owner of|PillPack|"Amazon’s deal to acquire One Medical follows its 2018 purchase of the online pharmacy service PillPack"|
|JPMorgan Chase|partner of|Amazon|"Amazon partnered with JPMorgan Chase and Berkshire Hathaway"|
|Berkshire Hathaway|partner of|Amazon|"Amazon partnered with JPMorgan Chase and Berkshire Hathaway"|
|Amazon|developer of|Haven|"Amazon partnered with JPMorgan Chase and Berkshire Hathaway on an effort to provide better health care services and insurance at a lower cost to workers and families at the three companies, and possibly other businesses, too. That effort, called Haven"|
|end|"""
    pprint(GPT3Utils.parse_entity_list(text))

    pprint(GPT3Utils.parse_table(text))

def main():
    # import asyncio
    # asyncio.run(test_promptinglmql())
    
    test_companyzeroshotclassificationpipeline()

if __name__ == "__main__":
    main()


