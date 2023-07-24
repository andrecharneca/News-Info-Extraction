import os
# get root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

ARTICLES_DF_PATH = os.path.join(ROOT_DIR, 'data/articles', 'articles.csv')
ARTICLES_JSONS_PATH = os.path.join(ROOT_DIR, 'data/articles')

MANUAL_ANNOTATIONS_DF_PATH = os.path.join(ROOT_DIR, 'data/manual_annotations', 'manual_annotations.csv')
MANUAL_ANNOTATIONS_JSONS_PATH = os.path.join(ROOT_DIR, 'data/manual_annotations')
MANUAL_ANNOTATIONS_RAW_DF_PATH = os.path.join(ROOT_DIR, 'data/raw', 'Business news articles - Manual Articles.csv')

TEMP_DATA_DIR = os.path.join(ROOT_DIR, 'data/temp')
RESULTS_DATA_DIR = os.path.join(ROOT_DIR, 'data/results')