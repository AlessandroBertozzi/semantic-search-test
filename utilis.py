import numpy as np


def get_bio_by_profile_id(data, profile_id):
    complete_sentence = [info_sentence['sentence'] for info_sentence in data if
                         info_sentence['profile_id'] == profile_id]
    return " ".join(complete_sentence)


def rebuild_bio(data):
    reorganized_bio = dict()
    bio_rebuilt = list()

    for sentence in data:

        if sentence['profile_id'] in reorganized_bio.keys():
            reorganized_bio[sentence['profile_id']] = " ".join(
                [reorganized_bio[sentence['profile_id']], sentence['sentence']])
        else:
            reorganized_bio[sentence['profile_id']] = sentence['sentence']

    for key, value in reorganized_bio.items():
        bio_rebuilt.append({
            "profile_id": key,
            "sentence": value
        })

    return bio_rebuilt


def rebuild_bio_with_vector(data):
    reorganized_bio = dict()
    bio_rebuilt = list()

    for sentence in data:

        if sentence['profile_id'] in reorganized_bio.keys():
            reorganized_bio[sentence['profile_id']] = np.mean(
                [reorganized_bio[sentence['profile_id']], sentence['vector']], axis=0)
        else:
            reorganized_bio[sentence['profile_id']] = sentence['vector']

    for key, value in reorganized_bio.items():
        bio_rebuilt.append({
            "profile_id": key,
            "vector": value
        })

    return bio_rebuilt


def rebuild_bio_with_list_sentences(data):
    reorganized_bio = dict()

    for sentence in data:

        if sentence['profile_id'] in reorganized_bio.keys():
            reorganized_bio[sentence['profile_id']].append(sentence['sentence'].strip())
        else:
            reorganized_bio[sentence['profile_id']] = [sentence['sentence'].strip()]

    return reorganized_bio
