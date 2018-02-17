import pandas as pd
from bs4 import BeautifulSoup as bs
import cache_df
from sklearn import svm
import math


columns_to_remove = ['level_0', 'index']

def drop_usless_columns(df):
    if (columns_to_remove[0] in df.columns) & \
            (columns_to_remove[1] in df.columns):
        return df.drop(columns_to_remove, axis=1)
    else:
        return df


def create_feature_column(df):
    feature_columns = [x for x in df.columns if str.isnumeric(str(x))]

    print(feature_columns)
    df["features"] = list(df[feature_columns].values)
    return df


def build_check_string(ra, rt, ca, ct):
    return "{0}_{1}_{2}_{3}".format(ra, rt, ca, ct)


def prepare_check_set(annotations):
    ra = [str.replace(x, ".xml", "") for x in annotations["Reference_Article"].values]
    rt = [(bs(x, 'xml').findAll("S")[0])["sid"] for x in annotations["Reference_Text"].values]

    ca = [str.replace(x, ".xml", "") for x in annotations["Citing_Article"].values]
    ct = [(bs(x, 'xml').findAll("S")[0])["sid"] for x in annotations["Citation_Text"].values]

    check_set = list()
    for i in range(0, len(ra)):
        check_set.append(build_check_string(ra[i], rt[i], ca[i], ct[i]))
    return set(check_set)


def check_label(r, check_set):

    if build_check_string(r["REF_doc"], r["REF_sid"], r["CIT_doc"], r["CIT_sid"]) in check_set:
        r["label"] = True
        return r
    else:
        r["label"] = False
        return r


def create_label_column(df, annotations):
    check_set = prepare_check_set(annotations)
    print(check_set)

    return df.apply(lambda r: check_label(r, check_set), axis=1)


def set_column_prefix(prefix, df):
    df.columns = map(lambda c: prefix + str(c), df.columns)
    return df



def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


def x_cleaning(r):
    print("r: ", r)

    r = str.replace(r, "[", "")
    r = str.replace(r, "]", "")
    r = str.replace(r,"\n", "")
    r = r.split(" ")
    print("r: ", r)

    r = [float(x) for x in r ]
    print(r)
    return r

def random_undersample(df, th=0.1):
    negative_df = df[df["label"] == 0]
    positive_df = df[df["label"] == 1]
    n_negative_sample = math.ceil(len(positive_df.index) * (1.0-th) / th)

    print("#positive_sample:", len(positive_df.index))
    print("#negative_sample:", n_negative_sample)

    negative_sample_df = negative_df.sample(n=n_negative_sample, replace=False)
    return pd.concat([positive_df, negative_sample_df], ignore_index=True)


def build_check_string_2(ca, ct):
    return "{0}_{1}".format(ca, ct)


def prepare_check_set_2(annotations):

    ca = [str.replace(x, ".xml", "") for x in annotations["Citing_Article"].values]
    ct = [(bs(x, 'xml').findAll("S")[0])["sid"] for x in annotations["Citation_Text"].values]

    check_set = list()
    for i in range(0, len(ca)):
        check_set.append(build_check_string_2(ca[i], ct[i]))
    return set(check_set)


def check_label_2(r, check_set):

    if build_check_string_2(r["CIT_doc"], r["CIT_sid"]) in check_set:
        r["label"] = True
        return r
    else:
        r["label"] = False
        return r


def random_undersample_cit(cit_scores_df, annotations):
    check_set = prepare_check_set_2(annotations)
    print("check set:", check_set)

    cit_scores_df = cit_scores_df.apply(lambda r: check_label_2(r, check_set), axis=1)

    count = cit_scores_df.groupby(["label"]).count()
    print(count)

    cit_scores_df = random_undersample(cit_scores_df, 0.5)
    #print(cit_scores_df)

    return cit_scores_df



def prepare_classification(ref_scores_df, cit_scores_df, annotations):
    c = cache_df.cache_df()

    ref_id = (ref_scores_df["doc"].values)[0]
    cache_file_name = ref_id + "_labeled_dataset.csv"
    print(cache_file_name)

    if c.check(cache_file_name):
        labeled_dataset = c.load_df(cache_file_name, lineterminator=";")

    else:

        ref_scores_df = set_column_prefix("REF_",
                                          create_feature_column(
                                              drop_usless_columns(ref_scores_df)))

        cit_scores_df = set_column_prefix("CIT_",
                                          create_feature_column(
                                              drop_usless_columns(cit_scores_df)))

        cit_scores_sampled_df = random_undersample_cit(cit_scores_df[["CIT_doc", "CIT_sid", "CIT_features"]], annotations)

        dataset = df_crossjoin(ref_scores_df[["REF_doc", "REF_sid", "REF_features"]],
                               cit_scores_sampled_df[["CIT_doc", "CIT_sid", "CIT_features"]]) \
            .sort_values(["REF_sid", "CIT_sid"])
#
        labeled_dataset = create_label_column(dataset, annotations)
#
        c.save_df(labeled_dataset, cache_file_name, index=False, line_terminator=";")


    return labeled_dataset


def merge_labeled_dataset_from_cache(ref_ids):
    global_ref_labeled_dataset = []
    c = cache_df.cache_df()


    for ref_id in ref_ids:
        try:

            cache_file_name = ref_id + "_labeled_dataset.csv"
            labeled_dataset = c.load_df(cache_file_name, lineterminator=";")
            global_ref_labeled_dataset.append(labeled_dataset)
        except Exception:
            print("skip:", ref_id)

    return pd.concat(global_ref_labeled_dataset, ignore_index=True)

def classification_train(labeled_dataset):
    c = cache_df.cache_df()

    labeled_dataset = random_undersample(labeled_dataset, th=0.1)

    count = labeled_dataset.groupby(["label"]).count()
    print("Count", count)

    values_x_temp = (labeled_dataset["REF_features"] + labeled_dataset["CIT_features"]).values
    type(values_x_temp)
    print(values_x_temp)

    values_x = []
    # i=0
    for x in values_x_temp:
        # i +=1
        # print(i,x)
        x = str.replace(x, "[", "")
        x = str.replace(x, "]", "")
        x = str.replace(x, "\n", "")
        x = x.split(" ")
        # print("r1: ", x)
        x = [float(y) for y in x if len(y) > 1]
        # print("r2: ", x)
        values_x.append(x)

    X = values_x  # map(x_cleaning, values_x)

    Y = labeled_dataset["label"].values

    Y = [bool(y) for y in Y]

    print("Y",Y)


    model_name = "global_svm_model.pkl"
    print(model_name)

    if c.check(model_name):
        clf = c.load_scikit_model(model_name)

    else:
        clf = svm.SVC(max_iter=100000, class_weight={True: 0.8, False: 0.2})  # class_weight={True: 0.8, False: 0.2}
        clf.fit(X, Y)
        c.save_scikit_model(clf, model_name)

    print(clf)



def classification(ref_scores_df, cit_scores_df, annotations):
    c = cache_df.cache_df()

    ref_id = (ref_scores_df["doc"].values)[0]
    cache_file_name = ref_id + "_labeled_dataset.csv"
    print(cache_file_name)


    if c.check(cache_file_name):
        labeled_dataset = c.load_df(cache_file_name, lineterminator=";")

    else:

        ref_scores_df = set_column_prefix("REF_",
                                          create_feature_column(
                                              drop_usless_columns(ref_scores_df)))

        cit_scores_df = set_column_prefix("CIT_",
                                          create_feature_column(
                                              drop_usless_columns(cit_scores_df)))

        dataset = df_crossjoin(ref_scores_df[["REF_doc", "REF_sid", "REF_features"]],
                               cit_scores_df[["CIT_doc", "CIT_sid", "CIT_features"]])\
            .sort_values(["REF_sid", "CIT_sid"])

        labeled_dataset = random_undersample(create_label_column(dataset, annotations))

        c.save_df(labeled_dataset, cache_file_name, index=False, line_terminator=";")

    #print(labeled_dataset)
    values_x_temp = (labeled_dataset["REF_features"] + labeled_dataset["CIT_features"]).values
    type(values_x_temp)
    print(values_x_temp)

    values_x = []
    #i=0
    for x in values_x_temp:
        #i +=1
        #print(i,x)
        x = str.replace(x, "[", "")
        x = str.replace(x, "]", "")
        x = str.replace(x, "\n", "")
        x = x.split(" ")
        #print("r1: ", x)
        x = [float(y) for y in x if len(y) > 1]
        #print("r2: ", x)
        values_x.append(x)

    X = values_x #map(x_cleaning, values_x)

    Y = labeled_dataset["label"]


    model_name = ref_id + "_svm_model.pkl"
    print(model_name)

    if c.check(model_name):
        clf = c.load_scikit_model(model_name)

    else:
        clf = svm.SVC(max_iter=10000) #class_weight={True: 0.8, False: 0.2}
        clf.fit(X, Y)
        c.save_scikit_model(clf, model_name)

    print(clf)

    return ""