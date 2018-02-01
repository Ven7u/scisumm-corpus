from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pandas as pd
from bs4 import BeautifulSoup
import nltk
import lxml
import os

import text_processing as tp

#input_folder = "/Users/francescoventura/Documents/POLI/Corsi/dottorato/text mining and analytics/exam/scisumm-corpus-master/"
input_folder = "../"
out_folder = "../out/"
data_folder = "data/"

def get_set_folder(type = "Training", year = "2017"):
    return  input_folder + data_folder + type + "-Set-" + year + "/"

def get_ref_ids():

    return ["C00-2123"]

def get_cit_ids():

    return ["C02-1050",
    "C04-1091",
    "E03-1007",
    "E06-1004",
    "H01-1062",
    "J03-1005",
    "J04-2003",
    "J04-4002",
    "N03-1010",
    "P01-1027",
    "P03-1039",
    "W01-0505",
    "W01-1404",
    "W01-1407",
    "W01-1408",
    "W02-1020"]




def load_ref_xml_source(set_folder,ref_id):
    return load_source(set_folder+ref_id+"/Reference_XML/"+ref_id+".xml",ref_id)


def load_cit_xml_source(set_folder, ref_id, cit_id):
    return load_source(set_folder+ref_id+"/Citance_XML/"+cit_id+".xml",cit_id)


def load_source(source, doc_name):
    file = source #+ str(id) + "/" + str(id) + "/DEFAULT/Part3DP.xml"  # sys.argv[1]
    handler = open(file).read()
    soup = BeautifulSoup(handler, 'xml')

    sentences = soup.findAll("S")
    sentences_dicts = []
    for s in sentences:
        sentences_a = {}
        sentences_a["doc"] = doc_name
        sentences_a["sid"] = s["sid"]
        #sentences_a["ssid"] = s["ssid"]
        sentences_a["tagged_text"] = s
        sentences_a["text"] = s.contents[0]
        sentences_dicts.append(sentences_a)
    return sentences_dicts

def load_ref_cit_tuple():
    return ""

def test_1():

    for ref_id in get_ref_ids():
        for cit_id in get_cit_ids():

            #s1 = load_source(source_1, "CIT_PAPER_1")
            #s2 = load_source(source_2, "REF_PAPER")

            s1 = load_ref_xml_source(get_set_folder(),ref_id)
            s2 = load_cit_xml_source(get_set_folder(),ref_id,cit_id)

            sentences_df = pd.DataFrame(data=s1 + s2)
            sentences_df["token_counts"] = sentences_df.apply(tp.tokens_count, axis=1)
            sentences_df = sentences_df[((sentences_df["token_counts"] > 10) & (sentences_df["token_counts"] <= 70))]

            print(sentences_df)



            vocabulary, tf = tp.preprocess(sentences_df)

            lsi_terms, lsi_rows = tp.run_lsi(tf,vocabulary,sentences_df)

            ref_scores, cit_scores = tp.get_ref_and_cit_scores(lsi_rows,ref_id, cit_id)

            cosine_scores = tp.get_cosine_similarities(ref_scores, cit_scores)

            result = tp.get_final_scores(cosine_scores, ref_scores, cit_scores)

            print(result)


            out_folder_p = out_folder + ref_id + "/" + cit_id + "/"
            out_file = "related_sentences.csv"

            if not os.path.exists(out_folder_p):
                os.makedirs(out_folder_p)

            result.to_csv(out_folder_p+out_file, index=False)



def main():
    test_1()

    return 0



if __name__ == '__main__':
    main()
