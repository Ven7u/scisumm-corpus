from collections import Counter
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import sklearn.metrics.pairwise as sklm
import numpy as np
import operator


stop_words = set(stopwords.words('english'))



def tokens_count(row):
    #print(row["text"])
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(row['text'])
    return len(tokens)

# print(stop_words)
def row_preprocess(row):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    # print(row)
    tokens = tokenizer.tokenize(row)
    # tokens = [nltk.word_tokenize(t) for t in tokens]
    # print(tokens)
    filtered_tokens = [t.lower() for t in tokens if t.lower() not in stop_words]
    # print(filtered_tokens)

    lmtzr = WordNetLemmatizer()
    lems = [lmtzr.lemmatize(t) for t in filtered_tokens]
    return " ".join(lems)



def preprocess(sentences_df):
    count_vect = CountVectorizer(stop_words="english", preprocessor=row_preprocess)
    X_train_counts = count_vect.fit_transform(sentences_df.text)
    # print(X_train_counts.shape)
    # print(count_vect.get_feature_names())
    vocabulary = pd.DataFrame(dict(count_vect.vocabulary_), index=["index"]).T.reset_index()
    vocabulary.columns = ["word", "index"]
    vocabulary = vocabulary.set_index("index")

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)

    return (vocabulary, X_train_tf)




def run_lsi(tf, vocabulary, sentences_df):


    svd = TruncatedSVD(n_components=20)
    svd_matrix_terms = svd.fit_transform(tf.T)
    svd_matrix_rows = svd.fit_transform(tf)

    print(svd_matrix_terms.shape)
    print(svd_matrix_rows.shape)

    lsi_terms = pd.DataFrame(svd_matrix_terms)
    lsi_terms = pd.merge(vocabulary, lsi_terms, left_index=True, right_index=True).reset_index().set_index(["index", "word"])


    lsi_rows = pd.DataFrame(svd_matrix_rows)
    lsi_rows = pd.merge(sentences_df[["doc", "sid", "text"]], lsi_rows, left_index=True,
                        right_index=True).reset_index().set_index(["index", "doc", "text"])
    return (lsi_terms, lsi_rows)


def get_ref_and_cit_scores(lsi_rows,ref_id,cit_id):
    scores = lsi_rows.reset_index()
    ref_scores = scores[scores["doc"] == ref_id].reset_index()
    ref_scores["sid"] = pd.to_numeric(ref_scores["sid"])

    cit_scores = scores[scores["doc"] == cit_id].reset_index()
    cit_scores["sid"] = pd.to_numeric(cit_scores["sid"])

    return (ref_scores, cit_scores)


def get_cosine_similarities(ref_scores,cit_scores):
    ref_ranks = ref_scores[["index", "doc", "text", "sid"]]
    cosine_scores = []

    x = ref_scores.iloc[:, 5:]
    y = cit_scores.iloc[:, 5:]

    print(x)
    print(y)
    cosine_scores = sklm.cosine_similarity(x, y)
    print(cosine_scores.shape)
    return cosine_scores


def get_final_scores(cosine_scores,ref_scores,cit_scores):


    similarity_max = []
    for r in range(len(cosine_scores)):
        cit_index, value = max(enumerate(cosine_scores[r]), key=operator.itemgetter(1))
        similarity_max.append(
            {"sid": ref_scores.iloc[r]["sid"], "cit_sid": cit_scores.iloc[cit_index]["sid"], "score": value})

    similarity_scores = pd.DataFrame(similarity_max)
    results = similarity_scores.merge(ref_scores[["doc", "sid", "text"]], on="sid").merge(cit_scores[["doc", "sid", "text"]],
                                                                                   left_on="cit_sid",
                                                                                   right_on="sid").sort_values("score",
                                                                                                               ascending=False)
    #print(results)
    results.columns = ["cit_sid","score","ref_sid","ref_doc","ref_text","cit_doc","cit_sid_2","cit_text"]
    results = results[["ref_doc","ref_sid","ref_text","cit_doc","cit_sid","cit_text","score"]]
    return results[0:15]


