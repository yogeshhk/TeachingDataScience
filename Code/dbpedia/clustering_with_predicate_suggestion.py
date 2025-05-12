# GSOC 2024 Abdulsobur https://colab.research.google.com/drive/1E0aIEagxPt9FLKn-8JbwI1WPjOwu9CbJ?usp=sharing

from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet
from funcy import print_durations
from gensim.models import KeyedVectors
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
from sklearn.cluster import KMeans
import time
import pandas as pd
import numpy as np
import os

NS_RESOURCE = 'http://dbpedia.org/resource/'
NS_RESOURCE_LEN = len(NS_RESOURCE)

NS_ONTOLOGY = 'http://dbpedia.org/ontology/'
NS_ONTOLOGY_LEN = len(NS_ONTOLOGY)


def to_uri(label, tbox):
    return list(filter(lambda x: 'A' <= x[NS_ONTOLOGY_LEN: NS_ONTOLOGY_LEN + 1] <= 'z', tbox[label]))


encoder_model = SentenceTransformer('all-mpnet-base-v2')


@print_durations
def get_embeddings(labels, sent_tran_model):
    embeddings = sent_tran_model.encode(labels, show_progress_bar=False)
    return embeddings


def retrieve_tbox(lang='en', offset=0):
    sparql = SPARQLWrapper('http://dbpedia.org/sparql')
    query = f"""
    SELECT ?uri ?label {{
      ?uri a ?type ; rdfs:label ?label .
      values(?type) {{ (owl:Class) (rdf:Property) }}
      filter(lang(?label) = '{lang}' && regex(?uri, "http://dbpedia.org/ontology/"))
    }} LIMIT 10000 OFFSET {offset}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    tbox = {}
    for result in results['results']['bindings']:
        uri = result['uri']['value']
        label = result['label']['value']
        if label not in tbox:
            tbox[label] = set()
        tbox[label].add(uri)
    return tbox


def get_labels_and_tbox():
    offset = 0
    tbox = {}
    while True:
        tbox_chunk = retrieve_tbox(lang='en', offset=offset)
        if len(tbox_chunk) == 0:
            break
        offset += 10000
        for k, v in tbox_chunk.items():
            if k not in tbox:
                tbox[k] = set()
            tbox[k] = tbox[k].union(v)
    labels = [l.replace('\n', ' ') for l in tbox]
    return labels, tbox


def write_embeddings_to_file(embeddings, labels, filename):
    with open(filename, 'w', encoding='utf-8') as f_out:
        f_out.write(f"{len(labels)} {len(embeddings[0])}\n")
        for label, embedding in zip(labels, embeddings):
            f_out.write(f"{label.replace(' ', '_')} {' '.join([str(x) for x in embedding])}\n")
    print("Embeddings written to file successfully")


def load_gensim_model_from_file(filepath):
    model = KeyedVectors.load_word2vec_format(filepath, binary=False)
    return model


def cluster_embeddings(embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_, kmeans.labels_


def dbpedia_similarity_search(pred, gensim_model, sent_tran_model, labels, tbox, cluster_centers,
                              embeddings, cluster_labels, threshold=0.5):
    print("Predicting:", pred)
    pred_embedding = sent_tran_model.encode([pred], show_progress_bar=False)[0]

    pred_embedding = pred_embedding.reshape(-1)

    similarities = gensim_model.cosine_similarities(pred_embedding, cluster_centers)
    closest_cluster_idx = np.argmax(similarities)
    closest_similarity = similarities[closest_cluster_idx]

    ####****Threshold not determined yet****####

    if closest_similarity < threshold:
        print(f"No close cluster found for predicate '{pred}' with similarity above {threshold}.")
        # new_predicate = wordnet_sim_search(pred, gensim_model_wordnet, sent_tran_model)
        # print(f"Suggested predicate dbp:{new_predicate}")
        return "----"

    print(f"Closest cluster index: {closest_cluster_idx}")
    print(f"Similarity to closest cluster: {closest_similarity}")

    group_indices = np.where(cluster_labels == closest_cluster_idx)[0]
    group_embeddings = [embeddings[i] for i in group_indices]
    group_labels = [labels[i] for i in group_indices]

    # print("Labels in the closest cluster:", group_labels)

    result = gensim_model.most_similar(positive=[pred_embedding], topn=1)
    out = []

    for label, score in result:
        out.append({'label': label.replace('_', ' '), 'score': score})
    df = pd.DataFrame(out)
    df.insert(0, 'URIs', df['label'].map(lambda x: to_uri(x, tbox=tbox)))
    return df


def populate_dbpedia_embeddings(vector_store_filename="models/dbpedia-ontology.vectors"):
    start = time.time()

    labels, tbox = get_labels_and_tbox()
    predicate_embeddings = get_embeddings(labels, encoder_model)
    write_embeddings_to_file(predicate_embeddings, labels, vector_store_filename)

    end = time.time()
    print(f"Time taken to compute and write embeddings - {end - start} seconds")
    return labels, tbox, predicate_embeddings


if __name__ == "__main__":
    store_filename = "models/dbpedia-ontology.vectors"
    labels, tbox, embeddings = populate_dbpedia_embeddings(store_filename)
    cluster_centers, cluster_labels = cluster_embeddings(embeddings)

    if os.path.isfile(store_filename):
        gensim_model = load_gensim_model_from_file(store_filename)

        # pred = "perimeter of"
        pred = "date of birth"
        # pred = "border with"

        db_pred = dbpedia_similarity_search(pred, gensim_model, encoder_model, labels, tbox, cluster_centers,
                                            embeddings, cluster_labels)

        print(db_pred)
    else:
        print(f"{store_filename} is not populated")
