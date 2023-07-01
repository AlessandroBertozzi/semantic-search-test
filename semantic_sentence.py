import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import statistics
from utilis import rebuild_bio_with_list_sentences


def semantic_text_similarity(data, test_profile_id, model='all-MiniLM-L6-v2', type_operation="mean"):
    embedder = SentenceTransformer(model)
    final_result = dict()
    all_scores = dict()
    data = rebuild_bio_with_list_sentences(data)

    # Split QUERY and CORPUS
    sentences1 = data.pop(test_profile_id)
    corpus_info_sentences = data

    embeddings1 = embedder.encode(sentences1, convert_to_tensor=True)

    # CORPUS
    for key, bio in corpus_info_sentences.items():

        sentences2 = bio
        embeddings2 = embedder.encode(sentences2, convert_to_tensor=True)
        # Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        for query in cosine_scores:
            printmax(query)


        # all_scores[key] = cosine_scores
        # if type_operation == 'mean':
        #     final_result[key] = torch.mean(cosine_scores).item()
        # if type_operation == 'median':
        #     final_result[key] = np.median(cosine_scores)

    return all_scores, final_result


def semantic_search(data, test_profile_id, model='all-MiniLM-L6-v2', top_k=5):
    embedder = SentenceTransformer(model)
    all_scores = dict()

    # Split QUERY and CORPUS
    query_sentences = list()
    corpus_info_sentences = list()

    for sentence in data:
        if sentence['profile_id'] == test_profile_id:
            query_sentences.append(sentence['sentence'])
        else:
            corpus_info_sentences.append(sentence)

    # CORPUS
    corpus_profile_id_position = list()
    corpus_sentences = list()

    for profile_sentence in corpus_info_sentences:
        corpus_profile_id_position.append(profile_sentence['profile_id'])
        corpus_sentences.append(profile_sentence['sentence'])

    corpus = corpus_sentences
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # QUERY sentences:
    queries = query_sentences

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(top_k, len(corpus))
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
        idx_already_computed = list()

        for score, idx in zip(top_results[0], top_results[1]):
            print(f"sentence with profile_id {corpus_profile_id_position[idx]}")
            print(corpus_sentences[idx], "(Score: {:.4f})".format(score))

            if corpus_profile_id_position[idx] not in idx_already_computed:
                idx_already_computed.append(corpus_profile_id_position[idx])
                if corpus_profile_id_position[idx] in all_scores.keys():
                    all_scores[corpus_profile_id_position[idx]].append(score.item())
                else:
                    all_scores[corpus_profile_id_position[idx]] = [score.item()]

    for values in all_scores.values():
        while len(values) < len(query_sentences):
            values.append(0)

    final_result = dict()
    for key, value in all_scores.items():
        final_result[key] = round(statistics.median(value), 3)

    return all_scores, final_result


def m_sentences(data, test_profile_id, model='all-MiniLM-L6-v2', top_k=5, type_operation="mean"):
    embedder = SentenceTransformer(model)
    final_result = dict()

    # Split query and corpus
    query_sentences = list()
    corpus_info_sentences = list()

    for sentence in data:
        if sentence['profile_id'] == test_profile_id:
            query_sentences.append(sentence['sentence'])
        else:
            corpus_info_sentences.append(sentence)

    # built corpus
    corpus_vector_sentence = dict()

    for profile_sentence in corpus_info_sentences:
        if profile_sentence['profile_id'] in corpus_vector_sentence.keys():
            corpus_vector_sentence[profile_sentence['profile_id']].append(profile_sentence['sentence'])
        else:
            corpus_vector_sentence[profile_sentence['profile_id']] = [profile_sentence['sentence']]

    corpus_profile_id_position = list()
    corpus_sentences = list()

    for key, value in corpus_vector_sentence.items():
        corpus_profile_id_position.append(key)
        if type_operation == 'mean':
            corpus_sentences.append(torch.mean(embedder.encode(value, convert_to_tensor=True), dim=0))
        elif type_operation == 'median':
            corpus_sentences.append(torch.from_numpy(np.median(embedder.encode(value), axis=0)))
    corpus_embeddings = torch.stack(corpus_sentences)

    # Query sentences:
    queries = query_sentences

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(top_k, len(corpus_embeddings))

    query_embedding = embedder.encode(queries, convert_to_tensor=True)
    if type_operation == 'mean':
        query_embedding = torch.mean(query_embedding, dim=0)
    elif type_operation == 'median':
        query_embedding = torch.from_numpy(np.median(query_embedding, axis=0))

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    for score, idx in zip(top_results[0], top_results[1]):
        final_result[corpus_profile_id_position[idx]] = score.item()
    return final_result


def best_rank_sentence(data, test_profile_id, min_score, model='all-MiniLM-L6-v2'):

    embedder = SentenceTransformer(model)
    final_result = dict()
    # Split QUERY and CORPUS
    query_sentences = list()
    corpus_info_sentences = list()

    for sentence in data:
        if sentence['profile_id'] == test_profile_id:
            query_sentences.append(sentence['sentence'])
        else:
            corpus_info_sentences.append(sentence)

    # CORPUS
    corpus_profile_id_position = list()
    corpus_sentences = list()

    for profile_sentence in corpus_info_sentences:
        corpus_profile_id_position.append(profile_sentence['profile_id'])
        corpus_sentences.append(profile_sentence['sentence'])

    corpus = corpus_sentences
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # QUERY sentences:
    queries = query_sentences

    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=1)

        # print("\n\n======================\n\n")
        # print("Query:", query)
        # print("\nTop most similar sentence in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            # print(corpus[idx], "(Score: {:.4f})".format(score))
            if score.item() >= min_score:
                final_result[corpus_profile_id_position[idx.item()]] = round(score.item(), 3)
    return final_result
