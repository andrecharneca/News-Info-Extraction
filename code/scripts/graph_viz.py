import pandas as pd
import sys 
import os
sys.path.append("../src/")
from pipelines.paths import (ARTICLES_JSONS_PATH,
                              RESULTS_DATA_DIR,
                              TEMP_DATA_DIR,
                              MANUAL_ANNOTATIONS_JSONS_PATH)
from pipelines.utils.relationships import (UNIDIR_RELATIONSHIP_LIST,
                                            BIDIR_RELATIONSHIP_LIST)
from pipelines.datamodules.datasets import ParagraphDataset
from pipelines.utils.utils import DatasetUtils
from pyvis import network as net

## Graph options ## 
options = """
const options = {
  "nodes": {
    "borderWidthSelected": null,
    "opacity": 1,
    "font": {
      "size": 24,
      "face": "tahoma",
      "strokeWidth": 6
    },
    "size": 40
  },
  "edges": {
    "color": {
      "inherit": true
    },
    "font": {
      "size": 22,
      "face": "tahoma",
      "strokeWidth": 16,
      "align": "top"
    },
    "scaling": {
      "min": 33
    },
    "selfReferenceSize": null,
    "selfReference": {
      "angle": 0.7853981633974483
    },
    "smooth": false,
    "width": 3
  },
  "manipulation": {
    "enabled": true
  },
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -25000,
      "springLength": 500,
      "springConstant": 0.1
    },
    "minVelocity": 0.75
  }
}
"""
### Functions ###

def add_edge_pyvis(graph, relationship, article_id, uni_dir_rel_list = UNIDIR_RELATIONSHIP_LIST, bi_dir_rel_list = BIDIR_RELATIONSHIP_LIST):
    """
    Add edge to pyvis graph based on relationship.
    args:
        graph: pyvis graph to add edge to
        relationship: relationship to add
        article_id: article id of relationsh
        uni_dir_rel_list: list of unidirectional relationships
        bi_dir_rel_list: list of bidirectional relationships
    returns:
        None
    """
    if relationship['relationship'] in uni_dir_rel_list:
        arrows = {"to": True, "from": False}
    elif relationship['relationship'] in bi_dir_rel_list:
        arrows = {"to": True, "from": True}
    else:
        print("Relationship not found: ", relationship['relationship'])
        arrows = 'to'
        
    graph.add_edge(relationship['entity_1'], relationship['entity_2'], label=relationship['relationship'], arrows=arrows, article_id=article_id)

def create_article_graph(article: dict, graph: net.Network = None):
    """
    Create a pyvis graph from an article.
    args:
        article: article to create graph from. With keys: article_id, entities, relationships
    returns:
        pyvis graph
    """
    if graph is None:
        graph = net.Network(notebook=True, directed=True, height="1080px", width="100%")
        graph.barnes_hut()
    for entity in article['entities']:
        graph.add_node(entity['word'], label=entity['word'], title=entity['word']) 
    for relationship in article['relationships']:
        add_edge_pyvis(graph, relationship, article['article_id'])

    return graph

def create_all_graphs():
    zsc_predictions = DatasetUtils.load_jsons(folder = os.path.join(RESULTS_DATA_DIR, "zsc"),
                                        data_path = ARTICLES_JSONS_PATH,
                                        article_id_list = None,
                                        tokenizer_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                                        max_tokens = 1e6)
    for article in zsc_predictions:
        graph = create_article_graph(article)
        # graph.show_buttons()
        graph.set_options(options)
        graph.show(TEMP_DATA_DIR + f"/graphs/zsc_graph_article_id_{article['article_id']}.html")

def join_graphs():
    zsc_article_1 = DatasetUtils.load_jsons(folder = os.path.join(RESULTS_DATA_DIR, "zsc"),
                                        data_path = ARTICLES_JSONS_PATH,
                                        article_id_list = [39],
                                        tokenizer_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                                        max_tokens = 1e6)
    zsc_article_22 = DatasetUtils.load_jsons(folder = os.path.join(RESULTS_DATA_DIR, "zsc"),
                                        data_path = ARTICLES_JSONS_PATH,
                                        article_id_list = [26],
                                        tokenizer_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                                        max_tokens = 1e6)
    graph_1 = create_article_graph(zsc_article_1[0])
    graph_1_and_2 = create_article_graph(zsc_article_22[0], graph_1)
    graph_1_and_2.set_options(options)
    graph_1_and_2.show(TEMP_DATA_DIR + f"/graphs/zsc_graph_article_id_39_and_26.html")

def join_all_graphs():
    zsc_predictions = DatasetUtils.load_jsons(folder = os.path.join(RESULTS_DATA_DIR, "zsc"),
                                        data_path = ARTICLES_JSONS_PATH,
                                        article_id_list = None,
                                        tokenizer_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                                        max_tokens = 1e6)
    graph = None
    for article in zsc_predictions:
        graph = create_article_graph(article, graph)
    graph.set_options(options)
    # graph.show_buttons(filter_=['physics'])
    graph.show(TEMP_DATA_DIR + f"/graphs/zsc_graph_all_articles.html")

def main():
    # create_all_graphs()
    join_all_graphs()

if __name__ == "__main__":
    main()
