import pandas as pd
from bs4 import BeautifulSoup
import os
import text_processing as tp
import text_classification as tc

# input_folder = "/Users/francescoventura/Documents/POLI/Corsi/dottorato/text mining and analytics/exam/scisumm-corpus-master/"
input_folder = "../"
out_folder = "../out/"
data_folder = "data/"


def get_set_folder(type="Training", year="2017"):
    return input_folder + data_folder + type + "-Set-" + year + "/"


def get_ref_ids():
    return ["C00-2123",
            "D10-1083",
            "J96-3004",
            "P06-2124",
            "W03-0410",
            "C02-1025",
            "E03-1020",
            "J98-2005",
            "P98-1046",
            "W04-0213",
            "C04-1089",
            "E09-2008",
            "N01-1011",
            "P98-1081",
            "W08-2222",
            "C08-1098",
            "H05-1115",
            "N04-1038",
            "P98-2143",
            "W95-0104",
            "C10-1045",
            "H89-2014",
            "N06-2049",
            "X96-1048",
            "C90-2039",
            "I05-5011",
            "P05-1004",
            "C94-2154",
            "J00-3003",
            "P05-1053"
            ]


def get_cit_ids(ref_id, annotations):


    tmp = annotations[annotations["Reference_Article"] == (ref_id + ".xml")]
    print(tmp)


    cit_ids = (annotations)["Citing_Article"].unique()

    return map(lambda n: str.replace(str.replace(n, ".txt", ""), ".xml", ""), cit_ids)


def get_annotation_values(ann):
    ann_values = [a.split(":", 1)[1].strip() for a in ann if len(ann) > 0]
    # print(ann_values)
    return ann_values


def get_annotation_names(ann):
    ann_names = [str.replace(a.split(":", 1)[0].strip(), " ", "_") for a in ann if len(ann) > 0]
    return ann_names


def load_annotation(set_folder, ref_id):
    source = set_folder + ref_id + "/annotation/" + ref_id + ".ann.txt"
    with open(source) as ann:
        content = ann.readlines()
        annotation_names = get_annotation_names(content[0].strip().split(" | "))
        annotations = [get_annotation_values(l.strip().split(" | ")) for l in content if len(l.strip()) > 0]

    annotations = pd.DataFrame(annotations, columns=annotation_names)
    print("annotations: ", annotations.columns)

    return annotations


def load_ref_xml_source(set_folder, ref_id):
    return load_source(set_folder + ref_id + "/Reference_XML/" + ref_id + ".xml", ref_id)


def load_cit_xml_source(set_folder, ref_id, cit_id):
    return load_source(set_folder + ref_id + "/Citance_XML/" + cit_id + ".xml", cit_id)


def load_source(source, doc_name):
    file = source  # + str(id) + "/" + str(id) + "/DEFAULT/Part3DP.xml"  # sys.argv[1]
    handler = open(file).read()
    soup = BeautifulSoup(handler, 'xml')

    sentences = soup.findAll("S")
    sentences_dicts = []
    for s in sentences:
        sentences_a = {}
        sentences_a["doc"] = doc_name
        sentences_a["sid"] = s["sid"]
        # sentences_a["ssid"] = s["ssid"]
        sentences_a["tagged_text"] = s
        sentences_a["text"] = s.contents[0]
        sentences_dicts.append(sentences_a)
    return sentences_dicts


def load_ref_cit_tuple():
    return ""


def test_4():
    ref_paper_tokens = []
    global_cit_paper_tokens = []
    annotations_per_ref = {}

    ref_scores_collection = []
    cit_scores_collection = []

    for ref_id in get_ref_ids():
        print("ref_id", ref_id)

        try:
            annotations = load_annotation(get_set_folder(), ref_id)
            #print(annotations)

            annotations_per_ref[ref_id] = annotations

            s1 = load_ref_xml_source(get_set_folder(), ref_id)
            ref_sentences_df = pd.DataFrame(data=s1)
            ref_paper_tokens.append(ref_sentences_df)

            # print("//////////////////////////////////")
            # print(ref_sentences_df)
            cit_papers_tokens = []

            for cit_id in get_cit_ids(ref_id, annotations):
                print("cit_id", cit_id)
                try:
                    # s1 = load_source(source_1, "CIT_PAPER_1")
                    # s2 = load_source(source_2, "REF_PAPER")
                    s2 = load_cit_xml_source(get_set_folder(), ref_id, cit_id)
                    sentences_df = pd.DataFrame(data=s2)
                    sentences_df["token_counts"] = sentences_df.apply(tp.tokens_count, axis=1)
                    sentences_df = sentences_df[((sentences_df["token_counts"] > 10) & (sentences_df["token_counts"] <= 70))]

                    print(len(sentences_df.index))
                    cit_papers_tokens.append(sentences_df)
                except Exception:
                    print("skip cit_id:", cit_id)

            #print(papers_tokens)
            # cit_papers_tokens.append(ref_sentences_df)
            cit_sentences_df = pd.concat(cit_papers_tokens, ignore_index=True)
            #print(cit_sentences_df)

            global_cit_paper_tokens.append(cit_sentences_df)
        except Exception:
            print("skip:", ref_id)

    cit_ref_sentences_df = pd.concat(ref_paper_tokens + global_cit_paper_tokens, ignore_index=True)

    print("Running Preprocess")
    vocabulary, tf = tp.preprocess(cit_ref_sentences_df)
    print("Running LSA")
    lsi_terms, lsi_rows = tp.run_lsi(tf, vocabulary, cit_ref_sentences_df)
    #print(lsi_rows.reset_index())

    for ref_id in get_ref_ids():
        try:
            for cit_id in get_cit_ids(ref_id, annotations_per_ref[ref_id]):
                ref_scores, cit_scores = tp.get_ref_and_cit_scores(lsi_rows, ref_id, cit_id)
                ref_scores_collection.append(ref_scores)
                cit_scores_collection.append(cit_scores)
        except Exception:
            print("skip:", ref_id)

    cit_scores_df = pd.concat(cit_scores_collection)

    print("cit_scores_df: ", cit_scores_df.columns)
    print("cit_scores_df: ", cit_scores_df)
    cit_scores_df[["doc", "sid"]].sort_values(["doc", "sid"]).to_csv("./cit_scores_df_test", index=False)
    ref_scores_df = pd.concat(ref_scores_collection)
    ref_scores_df[["doc", "sid"]].sort_values(["doc", "sid"]).to_csv("./ref_scores_df_test", index=False)
    # print("ref_scores_df: ", ref_scores_df.columns)
    # tc.classification(ref_scores_df, cit_scores_df, annotations)


    # print(cit_scores_df)


def test_3():
    for ref_id in get_ref_ids():
        ref_scores_collection = []
        cit_scores_collection = []

        annotations = load_annotation(get_set_folder(), ref_id)
        print(annotations)

        s1 = load_ref_xml_source(get_set_folder(), ref_id)
        ref_sentences_df = pd.DataFrame(data=s1)
        print("//////////////////////////////////")
        print(ref_sentences_df)
        papers_tokens = []

        for cit_id in get_cit_ids(ref_id, annotations):
            # s1 = load_source(source_1, "CIT_PAPER_1")
            # s2 = load_source(source_2, "REF_PAPER")
            s2 = load_cit_xml_source(get_set_folder(), ref_id, cit_id)
            sentences_df = pd.DataFrame(data=s2)
            sentences_df["token_counts"] = sentences_df.apply(tp.tokens_count, axis=1)
            sentences_df = sentences_df[((sentences_df["token_counts"] > 10) & (sentences_df["token_counts"] <= 70))]

            print(len(sentences_df.index))
            papers_tokens.append(sentences_df)

        # print(papers_tokens)
        papers_tokens.append(ref_sentences_df)
        cit_ref_sentences_df = pd.concat(papers_tokens, ignore_index=True)
        print(cit_ref_sentences_df)

        vocabulary, tf = tp.preprocess(cit_ref_sentences_df)

        lsi_terms, lsi_rows = tp.run_lsi(tf, vocabulary, cit_ref_sentences_df)

        print(lsi_rows.reset_index())

        for cit_id in get_cit_ids(ref_id, annotations):
            ref_scores, cit_scores = tp.get_ref_and_cit_scores(lsi_rows, ref_id, cit_id)
            ref_scores_collection.append(ref_scores)
            cit_scores_collection.append(cit_scores)

        cit_scores_df = pd.concat(cit_scores_collection)
        print("cit_scores_df: ", cit_scores_df.columns)
        print("cit_scores_df: ", cit_scores_df)
        cit_scores_df[["doc", "sid"]].sort_values(["doc", "sid"]).to_csv("./cit_scores_df_test", index=False)
        ref_scores_df = ref_scores_collection[0]
        ref_scores_df[["doc", "sid"]].sort_values(["doc", "sid"]).to_csv("./ref_scores_df_test", index=False)

        # print("ref_scores_df: ", ref_scores_df.columns)

        tc.classification(ref_scores_df, cit_scores_df, annotations)


        # print(cit_scores_df)


def test_2():
    for ref_id in get_ref_ids():
        ref_scores_collection = []
        cit_scores_collection = []

        annotations = load_annotation(get_set_folder(), ref_id)
        print(annotations)

        for cit_id in get_cit_ids(ref_id, annotations):
            # s1 = load_source(source_1, "CIT_PAPER_1")
            # s2 = load_source(source_2, "REF_PAPER")

            s1 = load_ref_xml_source(get_set_folder(), ref_id)
            s2 = load_cit_xml_source(get_set_folder(), ref_id, cit_id)

            sentences_df = pd.DataFrame(data=s1 + s2)
            sentences_df["token_counts"] = sentences_df.apply(tp.tokens_count, axis=1)
            sentences_df = sentences_df[((sentences_df["token_counts"] > 10) & (sentences_df["token_counts"] <= 70))]

            print(sentences_df)

            vocabulary, tf = tp.preprocess(sentences_df)

            lsi_terms, lsi_rows = tp.run_lsi(tf, vocabulary, sentences_df)

            ref_scores, cit_scores = tp.get_ref_and_cit_scores(lsi_rows, ref_id, cit_id)
            ref_scores_collection.append(ref_scores)
            cit_scores_collection.append(cit_scores)

    cit_scores_df = pd.concat(cit_scores_collection)
    print("cit_scores_df: ", cit_scores_df.columns)
    cit_scores_df[["doc", "sid"]].sort_values(["doc", "sid"]).to_csv("./cit_scores_df_test", index=False)
    ref_scores_df = ref_scores_collection[0]
    ref_scores_df[["doc", "sid"]].sort_values(["doc", "sid"]).to_csv("./ref_scores_df_test", index=False)

    print("ref_scores_df: ", ref_scores_df.columns)

    tc.classification(ref_scores_df, cit_scores_df, annotations)


    # print(cit_scores_df)


def test_1():
    for ref_id in get_ref_ids():
        annotations = load_annotation(get_set_folder(), ref_id)
        for cit_id in get_cit_ids(ref_id, annotations):

            # s1 = load_source(source_1, "CIT_PAPER_1")
            # s2 = load_source(source_2, "REF_PAPER")

            s1 = load_ref_xml_source(get_set_folder(), ref_id)
            s2 = load_cit_xml_source(get_set_folder(), ref_id, cit_id)

            sentences_df = pd.DataFrame(data=s1 + s2)
            sentences_df["token_counts"] = sentences_df.apply(tp.tokens_count, axis=1)
            sentences_df = sentences_df[((sentences_df["token_counts"] > 10) & (sentences_df["token_counts"] <= 70))]

            print(sentences_df)

            vocabulary, tf = tp.preprocess(sentences_df)

            lsi_terms, lsi_rows = tp.run_lsi(tf, vocabulary, sentences_df)

            ref_scores, cit_scores = tp.get_ref_and_cit_scores(lsi_rows, ref_id, cit_id)

            cosine_scores = tp.get_cosine_similarities(ref_scores, cit_scores)

            result = tp.get_final_scores(cosine_scores, ref_scores, cit_scores)

            print(result)

            out_folder_p = out_folder + ref_id + "/" + cit_id + "/"
            out_file = "related_sentences.csv"

            if not os.path.exists(out_folder_p):
                os.makedirs(out_folder_p)

            result.to_csv(out_folder_p + out_file, index=False)


def main():
    # test_2() # deprecated
    test_3() # modello lsa locale
    #test_4() # modello lsa globale

    return 0


if __name__ == '__main__':
    main()
