from dotenv import load_dotenv
import os
from elasticsearch import Elasticsearch

# take environment variables from .env.
load_dotenv()

# dev db
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
PROJECT_INDEX = os.getenv("PROJECT_INDEX")
PROFILE_INDEX = os.getenv("PROFILE_INDEX")

# dev database connection
dev_client = Elasticsearch(
    DB_HOST + ":" + DB_PORT,
    basic_auth=(DB_USER, DB_PASS),
    verify_certs=True,
    request_timeout=10
)


def run_profile():
    result = list()

    query = {
        "match_all": {
        }
    }

    search_results = dev_client.search(index=PROFILE_INDEX, query=query, scroll="30s", size=10000)['hits']['hits']

    for single_result in search_results:
        result.append({
            'profile_id': single_result['_source']['profile_id'],
            'sentence': single_result['_source']['sentence'],
            'n_sentence': single_result['_source']['n_sentence']
        })

    return result


def run_project():
    result = list()

    query = {
        "match_all": {
        }
    }

    search_results = dev_client.search(index=PROJECT_INDEX, query=query, scroll="30s", size=10000)['hits']['hits']

    for single_result in search_results:
        result.append({
            'profile_id': single_result['_source']['project_id'],
            'sentence': single_result['_source']['sentence'],
            'n_sentence': single_result['_source']['n_sentence']
        })

    return result
