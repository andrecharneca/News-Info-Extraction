import sys
from tqdm import tqdm
import pandas as pd
from time import time
import os
import numpy as np
from pprint import pprint

sys.path.append("../src/")
from pipelines.paths import (ARTICLES_JSONS_PATH,
                              RESULTS_DATA_DIR,
                              MANUAL_ANNOTATIONS_JSONS_PATH)
from pipelines.datamodules.datasets import ParagraphDataset
from pipelines.utils.utils import DatasetUtils, MetricsUtils, EntityUtils
from run_pipelines import run_zsc_pipeline


article_id_list = None
zero_shot_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
best_zsc_path = os.path.join(RESULTS_DATA_DIR, "zsc_hyperparam_search/max_tokens=128_zsc_threshold=0.995_ner_threshold=0.995")
device = 0


def load_datasets():
    zsc_predictions = DatasetUtils.load_jsons(folder = best_zsc_path,
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

    return zsc_predictions, gpt_predictions, target

def show_article_metrics(zsc_predictions, gpt_predictions, target, id):
    target_article = ParagraphDataset(data=target.get_filtered_by_key_vals("article_id", id))
    gpt_article = ParagraphDataset(data=gpt_predictions.get_filtered_by_key_vals("article_id", id))
    zsc_article = ParagraphDataset(data=zsc_predictions.get_filtered_by_key_vals("article_id", id))

    print("\n-Target-\n")
    for e in target_article[0]["entities"]:
        print(f"{e['word']}", end=", ")
    print()
    for relationship in target_article[0]["relationships"]:
        print(f"{relationship['entity_1']},{relationship['relationship']},{relationship['entity_2']}")

    print("\n-ZSC-\n")
    if len(zsc_article) > 0:
        for e in zsc_article[0]["entities"]:
            print(f"{e['word']} ({e['entity_group']})", end=", ")
        print()
        for relationship in zsc_article[0]["relationships"]:
            print(f"{relationship['entity_1']},{relationship['relationship']},{relationship['entity_2']}, {np.mean(relationship['scores']):.4f}")

    print("\n-GPT3-\n")
    if len(gpt_article) > 0:
        for e in gpt_article[0]["entities"]:
            print(f"{e['word']} ({e['entity_group']})", end=", ")
        print()
        for relationship in gpt_article[0]["relationships"]:
            print(f"{relationship['entity_1']},{relationship['relationship']},{relationship['entity_2']}, {relationship['passage']}")

    print("\n-Metrics-\n")
    print(f"ZSC: {MetricsUtils.compute_avg_article_metrics(zsc_article, target_article)}")
    print(f"GPT3: {MetricsUtils.compute_avg_article_metrics(gpt_article, target_article)}")

def metrics_report():
    zsc_predictions, gpt_predictions, target = load_datasets()
    article_ids = target.get_key_vals("article_id")
    print("Article IDs:", article_ids)
    for id in article_ids:
        print(f"\n---Article {id}---\n")
        print("URL:", target.get_filtered_by_key_vals("article_id", id)[0]["url"])
        show_article_metrics(zsc_predictions, gpt_predictions, target, id)
    
    print("\n---Total---\n")    
    total_metrics_zsc = MetricsUtils.compute_avg_article_metrics(zsc_predictions, target)
    total_metrics_gpt = MetricsUtils.compute_avg_article_metrics(gpt_predictions, target)
    print(f"ZSC: {total_metrics_zsc}")
    print(f"GPT3: {total_metrics_gpt}")

def compute_gpt3_entity_metrics():
    _, gpt_predictions, target = load_datasets()
    total_metrics_gpt = MetricsUtils.compute_avg_article_metrics(gpt_predictions, target)
    print(f"GPT3: {total_metrics_gpt}")

def zsc_hyperparam_search():
    hyperparam_lists = {
        "max_tokens": [128, 256],#[0, 64, 128, 256],
        "zsc_threshold": [0.9, 0.95, 0.99, 0.995, 0.999],
        "ner_threshold": [0.9, 0.95, 0.99, 0.995, 0.999],
    }
    target = DatasetUtils.load_jsons(folder = MANUAL_ANNOTATIONS_JSONS_PATH,
                                            data_path = ARTICLES_JSONS_PATH,
                                            article_id_list = article_id_list,
                                            tokenizer_model_name = zero_shot_model_name,
                                            max_tokens = 1e6)
    DatasetUtils.order_bidir_relationships_entities(target)
    # df_metrics = pd.DataFrame(columns=["max_tokens", "zsc_threshold", "ner_threshold", "relationship_precision", "relationship_recall", "relationship_f1", "entity_precision", "entity_recall", "entity_f1"])
    # load csv
    df_metrics = pd.read_csv(os.path.join(RESULTS_DATA_DIR, "zsc_hyperparam_search/metrics.csv"))

    for max_tokens in tqdm(hyperparam_lists["max_tokens"]):
        for zsc_threshold in hyperparam_lists["zsc_threshold"]:
            for ner_threshold in hyperparam_lists["ner_threshold"]:
                print(f"\n---max_tokens: {max_tokens}, zsc_threshold: {zsc_threshold}, ner_threshold: {ner_threshold}---\n")
                folder_name = f"zsc_hyperparam_search/max_tokens={max_tokens}_zsc_threshold={zsc_threshold}_ner_threshold={ner_threshold}"

                if not os.path.exists(os.path.join(RESULTS_DATA_DIR, folder_name)):
                    run_zsc_pipeline(max_tokens = max_tokens, zsc_threshold = zsc_threshold, ner_threshold = ner_threshold, folder_name = folder_name, verbose=False, device=device)
                    zsc_predictions = DatasetUtils.load_jsons(folder = os.path.join(RESULTS_DATA_DIR, folder_name),
                                                                data_path = ARTICLES_JSONS_PATH,
                                                                article_id_list = None,
                                                                tokenizer_model_name = zero_shot_model_name,
                                                                max_tokens = max_tokens)
                    total_metrics_zsc = MetricsUtils.compute_avg_article_metrics(zsc_predictions, target)
                    df_metrics = pd.concat([df_metrics, pd.DataFrame([[max_tokens, zsc_threshold, ner_threshold, 
                                                                    total_metrics_zsc["relationships"]["precision"], total_metrics_zsc["relationships"]["recall"], total_metrics_zsc["relationships"]["f1"], 
                                                                    total_metrics_zsc["entities"]["precision"], total_metrics_zsc["entities"]["recall"], total_metrics_zsc["entities"]["f1"]]], 
                                                                    columns=df_metrics.columns)], ignore_index=True)
                    print(f"Metrics: {total_metrics_zsc}")
                    df_metrics.to_csv(os.path.join(RESULTS_DATA_DIR, "zsc_hyperparam_search/metrics.csv"), index=False)

def analyze_metrics():
    df_metrics = pd.read_csv(os.path.join(RESULTS_DATA_DIR, "zsc_hyperparam_search/metrics.csv"))
    df_metrics = df_metrics.sort_values(by=["relationship_f1"], ascending=False)
    df_metrics.to_csv(os.path.join(RESULTS_DATA_DIR, "zsc_hyperparam_search/metrics_sorted_relationship_f1.csv"), index=False)

    df_metrics = df_metrics.sort_values(by=["entity_f1"], ascending=False)
    df_metrics.to_csv(os.path.join(RESULTS_DATA_DIR, "zsc_hyperparam_search/metrics_sorted_entity_f1.csv"), index=False)
    
    #show
    print(df_metrics)

def main():
    metrics_report()
    

if __name__ == "__main__":
    main()
