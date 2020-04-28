#  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
import string
from fastai.text import *  # just for utilty functions pd, np, Path etc.

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ...helpers.training import set_seed

def transform_df(df):
    df=df.replace(re.compile(r"(xxref|xxanchor)-[\w\d-]*"), "\\1 ")
    df=df.replace(re.compile(r"(^|[ ])\d+\.\d+\b"), " xxnum ")
    df=df.replace(re.compile(r"(^|[ ])\d\b"), " xxnum ")
    df=df.replace(re.compile(r"\bdata set\b"), " dataset ")
    df = df.drop_duplicates(["text", "cell_content", "cell_type"]).fillna("")
    return df

def train_valid_split(df, seed=42, by="cell_content"):
    set_seed(seed, "val_split")
    contents = np.random.permutation(df[by].unique())
    val_split = int(len(contents)*0.1)
    val_keys = contents[:val_split]
    split = df[by].isin(val_keys)
    valid_df = df[split]
    train_df = df[~split]
    len(train_df), len(valid_df)
    return train_df, valid_df

def get_class_column(y, classIdx):
    if len(y.shape) == 1:
        return y == classIdx
    else:
        return y.iloc[:, classIdx]

def get_number_of_classes(y):
    if len(y.shape) == 1:
        return len(np.unique(y))
    else:
        return y.shape[1]

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
re_tok_fixed = re.compile(
    f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])'.replace('<', '').replace('>', '').replace('/', ''))

def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

def tokenize_fixed(s):
    return re_tok_fixed.sub(r' \1 ', s).split()


class NBSVM:
    def __init__(self, experiment):
        self.experiment = experiment


    def pr(self, y_i, y):
        p = self.trn_term_doc[y == y_i].sum(0)
        return (p+1) / ((y == y_i).sum()+1)

    def get_mdl(self, y):
        y = y.values
        r = np.log(self.pr(1, y) / self.pr(0, y))
        m = LogisticRegression(C=self.experiment.C, penalty=self.experiment.penalty,
                               dual=self.experiment.dual, solver=self.experiment.solver,
                               max_iter=self.experiment.max_iter, class_weight=self.experiment.class_weight)
        x_nb = self.trn_term_doc.multiply(r)
        return m.fit(x_nb, y), r

    def bow(self, X_train):
        self.n = X_train.shape[0]

        tokenizer = tokenize_fixed if self.experiment.fixed_tokenizer else tokenize
        if self.experiment.vectorizer == "tfidf":
            self.vec = TfidfVectorizer(ngram_range=self.experiment.ngram_range,
                                       tokenizer=tokenizer,
                                       lowercase=self.experiment.lowercase,
                                       analyzer=self.experiment.analyzer,
                                       min_df=self.experiment.min_df, max_df=self.experiment.max_df,
                                       strip_accents='unicode', use_idf=1,
                                       smooth_idf=1, sublinear_tf=1)
        elif self.experiment.vectorizer == "count":
            self.vec = CountVectorizer(ngram_range=self.experiment.ngram_range, tokenizer=tokenizer,
                                       analyzer=self.experiment.analyzer,
                                       lowercase=self.experiment.lowercase,
                                       min_df=self.experiment.min_df, max_df=self.experiment.max_df,
                                       strip_accents='unicode')
        else:
            raise Exception(f"Unknown vectorizer type: {self.experiment.vectorizer}")

        return self.vec.fit_transform(X_train)

    def train_models(self, y_train):
        self.models = []
        if self.experiment.multinomial_type == "manual":
            for i in range(0, self.c):
                #print('fit', i)
                m, r = self.get_mdl(get_class_column(y_train, i))
                self.models.append((m, r))
        elif self.experiment.multinomial_type == "multinomial" or self.experiment.multinomial_type == "ovr":
            m = LogisticRegression(C=self.experiment.C, penalty=self.experiment.penalty,
                                   dual=self.experiment.dual, solver=self.experiment.solver,
                                   max_iter=self.experiment.max_iter,
                                   multi_class=self.experiment.multinomial_type, class_weight=self.experiment.class_weight)
            x_nb = self.trn_term_doc
            self.models.append(m.fit(x_nb, y_train))
        else:
            raise Exception(f"Unsupported multinomial_type {self.experiment.multinomial_type}")

    def fit(self, X_train, y_train):
        self.trn_term_doc = self.bow(X_train)
        self.c = get_number_of_classes(y_train)
        self.train_models(y_train)

    def predict_proba(self, X_test):
        test_term_doc = self.vec.transform(X_test)
        if self.experiment.multinomial_type == "manual":
            preds = np.zeros((len(X_test), self.c))
            for i in range(0, self.c):
                m, r = self.models[i]
                preds[:, i] = m.predict_proba(test_term_doc.multiply(r))[:, 1]
        elif self.experiment.multinomial_type == "multinomial" or self.experiment.multinomial_type == "ovr":
            preds = self.models[0].predict_proba(test_term_doc)
        else:
            raise Exception(f"Unsupported multinomial_type {self.experiment.multinomial_type}")
        return preds

    def sort_features_by_importance(self, label):
        label = label.value
        names = np.array(self.vec.get_feature_names())
        if self.experiment.multinomial_type == "manual":
            m, r = self.models[label]
            f = m.coef_[0] * np.array(r)[0]
        elif self.experiment.multinomial_type == "multinomial":
            f = self.models[0].coef_[label]
        else:
            raise Exception(f"Unsupported multinomial_type {self.experiment.multinomial_type}")
        if self.experiment.vectorizer == "tfidf":
            f *= self.vec.idf_
        indices = f.argsort()[::-1]
        return names[indices], f[indices]

    def get_mismatched(self, df, true_label, predicted_label):
        if self.experiment.merge_fragments and self.experiment.merge_type != "concat":
            print("warning: the returned results are before merging")
        true_label = true_label.value
        predicted_label = predicted_label.value

        probs = self.predict_proba(df["text"])
        preds = np.argmax(probs, axis=1)
        true_y = df["label"]

        mismatched_indices = (true_y == true_label) & (preds == predicted_label)
        mismatched = df[mismatched_indices]
        diff = probs[mismatched_indices, true_label] - probs[mismatched_indices, predicted_label]
        indices = diff.argsort()
        mismatched = mismatched.iloc[indices]
        mismatched["pr_diff"] = diff[indices]
        return mismatched

    def validate(self, X_test, y_test):
        acc = (np.argmax(self.predict_proba(X_test),  axis=1) == y_test).mean()
        return acc

def metrics(preds, true_y):
    y = true_y
    p = preds
    acc = (p == y).mean()
    tp = ((y != 0) & (p == y)).sum()
    fp = ((p != 0) & (p != y)).sum()
    fn = ((y != 0) & (p == 0)).sum()

    prec = tp / (fp + tp)
    reca = tp / (fn + tp)
    return {
        "precision": prec,
        "accuracy": acc,
        "recall": reca,
        "TP": tp,
        "FP": fp,
    }


def preds_for_cell_content(test_df, probs, group_by=["cell_content"]):
    test_df = test_df.copy()
    test_df["pred"] = np.argmax(probs, axis=1)
    grouped_preds = test_df.groupby(group_by)["pred"].agg(
        lambda x: x.value_counts().index[0])
    grouped_counts = test_df.groupby(group_by)["pred"].count()
    results = pd.DataFrame({'true': test_df.groupby(group_by)["label"].agg(lambda x: x.value_counts().index[0]),
                            'pred': grouped_preds,
                            'counts': grouped_counts})
    return results

def preds_for_cell_content_multi(test_df, probs, group_by=["cell_content"]):
    test_df = test_df.copy()
    probs_df = pd.DataFrame(probs, index=test_df.index)
    test_df = pd.concat([test_df, probs_df], axis=1)
    grouped_preds = np.argmax(test_df.groupby(
        group_by)[probs_df.columns].sum().values, axis=1)
    grouped_counts = test_df.groupby(group_by)["label"].count()
    results = pd.DataFrame({'true': test_df.groupby(group_by)["label"].agg(lambda x: x.value_counts().index[0]),
                            'pred': grouped_preds,
                            'counts': grouped_counts})
    return results

def preds_for_cell_content_max(test_df, probs, group_by=["cell_content"]):
    test_df = test_df.copy()
    probs_df = pd.DataFrame(probs, index=test_df.index)
    test_df = pd.concat([test_df, probs_df], axis=1)
    grouped_preds = np.argmax(test_df.groupby(
        group_by)[probs_df.columns].max().values, axis=1)
    grouped_counts = test_df.groupby(group_by)["label"].count()
    results = pd.DataFrame({'true': test_df.groupby(group_by)["label"].agg(lambda x: x.value_counts().index[0]),
                            'pred': grouped_preds,
                            'counts': grouped_counts})
    return results

def test_model(model, tdf):
    probs = model(tdf["text"])
    preds = np.argmax(probs, axis=1)
    print("Results of categorisation on text fagment level")
    print(metrics(preds, tdf.label))

    print("Results per cell_content grouped using majority voting")
    results = preds_for_cell_content(tdf, probs)
    print(metrics(results["pred"], results["true"]))

    print("Results per cell_content grouped with multi category mean")
    results = preds_for_cell_content_multi(tdf, probs)
    print(metrics(results["pred"], results["true"]))

    print("Results per cell_content grouped with multi category mean - only on fragments from the same paper that the coresponding table")
    results = preds_for_cell_content_multi(
        tdf[tdf.this_paper], probs[tdf.this_paper])
    print(metrics(results["pred"], results["true"]))
