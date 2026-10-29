"""Models with one interface.

    m = make(name).fit(train_texts, y_train, val_texts, y_val)   # tunes on val only
    p = m.predict_proba(texts)                                    # Platt-calibrated on val

Every featuriser is fit on the source TRAINING split only.
"""
import re, gzip, string
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import Platt
import config as C

GRID = [0.25, 1.0, 4.0]
MAXCH = 20000                     # characters kept per document


def clip(texts):
    return [str(t)[:MAXCH] for t in texts]


class Base:
    def _select(self, fits, Xv, yv):
        best = max(fits, key=lambda m: roc_auc_score(yv, self._raw(m, Xv)))
        return best

    def predict_proba(self, texts):
        return self.cal(self._raw(self.m, self.feat(texts)))


class TfidfLinear(Base):
    def __init__(self, kind="lr", ngram=(1, 2), max_features=100000, analyzer="word",
                 vocab_drop=None):
        self.kind, self.ngram, self.maxf, self.analyzer = kind, ngram, max_features, analyzer
        self.vocab_drop = set(vocab_drop) if vocab_drop is not None else set()

    def feat(self, texts):
        X = self.vec.transform(clip(texts))
        return X[:, self.keep] if self.keep is not None else X

    def _raw(self, m, X):
        return m.decision_function(X)

    def fit(self, tx, y, vx, yv, sample_weight=None):
        self.vec = TfidfVectorizer(ngram_range=self.ngram, min_df=2, max_features=self.maxf,
                                   sublinear_tf=True, analyzer=self.analyzer,
                                   strip_accents=None, lowercase=True)
        X = self.vec.fit_transform(clip(tx))
        self.keep = None
        if self.vocab_drop:
            vocab = self.vec.get_feature_names_out()
            self.keep = np.array([i for i, w in enumerate(vocab) if w not in self.vocab_drop])
            X = X[:, self.keep]
        Xv = self.feat(vx)
        fits = []
        for c in GRID:
            if self.kind == "lr":
                m = LogisticRegression(C=c, max_iter=2000, solver="liblinear")
            else:
                m = LinearSVC(C=c / 4, max_iter=5000)
            fits.append(m.fit(X, y, sample_weight=sample_weight))
        self.m = self._select(fits, Xv, yv)
        self.cal = Platt().fit(self._raw(self.m, Xv), yv)
        return self

    def feature_names(self):
        names = self.vec.get_feature_names_out()
        return names[self.keep] if self.keep is not None else names


class NBSVM(Base):
    """Wang & Manning (2012) NB-weighted logistic regression on binary n-grams."""
    def feat(self, texts):
        return self.vec.transform(clip(texts)).multiply(self.r).tocsr()

    def _raw(self, m, X):
        return m.decision_function(X)

    def fit(self, tx, y, vx, yv, sample_weight=None):
        self.vec = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, binary=True)
        X = self.vec.fit_transform(clip(tx))
        y = np.asarray(y)
        p = np.asarray(X[y == 1].sum(0)).ravel() + 1
        q = np.asarray(X[y == 0].sum(0)).ravel() + 1
        self.r = sparse.csr_matrix(np.log((p / p.sum()) / (q / q.sum())))
        Xr, Xv = X.multiply(self.r).tocsr(), self.feat(vx)
        fits = [LogisticRegression(C=c, max_iter=2000, solver="liblinear").fit(Xr, y) for c in GRID]
        self.m = self._select(fits, Xv, yv)
        self.cal = Platt().fit(self._raw(self.m, Xv), yv)
        return self


_GLOVE = None


def glove():
    global _GLOVE
    if _GLOVE is None:
        words, vecs = {}, []
        with gzip.open(C.GLOVE, "rt", encoding="utf8") as f:
            first = f.readline().split()
            if len(first) != 2:                      # no header line
                f.seek(0)
            for line in f:
                parts = line.rstrip().split(" ")
                words[parts[0]] = len(vecs)
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
        _GLOVE = (words, np.vstack(vecs))
    return _GLOVE


TOK = re.compile(r"[a-z0-9']+")


class GloveLR(Base):
    def feat(self, texts):
        words, M = glove()
        out = np.zeros((len(texts), M.shape[1]), np.float32)
        for i, t in enumerate(texts):
            ids = [words[w] for w in TOK.findall(str(t)[:MAXCH].lower()) if w in words]
            if ids:
                out[i] = M[ids].mean(0)
        return out

    def _raw(self, m, X):
        return m.decision_function(X)

    def fit(self, tx, y, vx, yv, sample_weight=None):
        X, Xv = self.feat(tx), self.feat(vx)
        fits = [LogisticRegression(C=c, max_iter=3000).fit(X, y) for c in GRID]
        self.m = self._select(fits, Xv, yv)
        self.cal = Platt().fit(self._raw(self.m, Xv), yv)
        return self


# ------------------------------------------------------------------ style / metadata only
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those he she they we you i me my our your his her their them us
not no so than then there here what which who whom whose when where why how all any each few
more most other some such only own same too very can will just should now""".split())
WEEKDAYS = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
DATELINE = re.compile(r"^\s*[A-Z][A-Z .,'-]{2,40}\s*(\([^)]{2,25}\))?\s*[-—]")
STYLE_NAMES = ["n_chars", "n_words", "mean_word_len", "type_token", "n_sent", "excl", "quest",
               "quotes", "upper_ratio", "allcaps_words", "digit_ratio", "stop_ratio",
               "first_person", "second_person", "weekday_mentions", "said_rate", "dateline",
               "urls", "mentions", "hashtags", "parentheses", "colon_semicolon"]


_STYLE_CACHE = {}


def style_features(texts):
    """Cached per document (features depend only on the text)."""
    out = []
    for t in texts:
        k = hash(str(t)[:MAXCH])
        if k not in _STYLE_CACHE:
            _STYLE_CACHE[k] = _style_one(str(t))
        out.append(_STYLE_CACHE[k])
    return np.asarray(out, np.float32)


def _style_one(t):
    rows = []
    for t in [t]:
        t = str(t)[:MAXCH]
        w = t.split()
        n = max(len(w), 1)
        lw = [x.lower().strip(string.punctuation) for x in w]
        rows.append([
            len(t), len(w), np.mean([len(x) for x in w]) if w else 0, len(set(lw)) / n,
            max(1, len(re.findall(r"[.!?]+(\s|$)", t))), t.count("!") / n, t.count("?") / n,
            (t.count('"') + t.count("“") + t.count("”")) / n,
            sum(ch.isupper() for ch in t) / max(len(t), 1),
            sum(1 for x in w if len(x) > 2 and x.isupper()) / n,
            sum(ch.isdigit() for ch in t) / max(len(t), 1),
            sum(1 for x in lw if x in STOP) / n,
            sum(1 for x in lw if x in {"i", "me", "my", "we", "our", "us"}) / n,
            sum(1 for x in lw if x in {"you", "your", "yours"}) / n,
            len(WEEKDAYS.findall(t)) / n, sum(1 for x in lw if x == "said") / n,
            1.0 if DATELINE.match(t) else 0.0,
            len(re.findall(r"https?://|\bURL\b", t)) / n, t.count("@") / n, t.count("#") / n,
            t.count("(") / n, (t.count(":") + t.count(";")) / n])
    return rows[0]


class StyleGB(Base):
    def feat(self, texts):
        return style_features(texts)

    def _raw(self, m, X):
        return m.predict_proba(X)[:, 1]

    def fit(self, tx, y, vx, yv, sample_weight=None):
        X, Xv = self.feat(tx), self.feat(vx)
        fits = [HistGradientBoostingClassifier(max_iter=it, learning_rate=0.1, random_state=0)
                .fit(X, y) for it in (100, 300)]
        self.m = self._select(fits, Xv, yv)
        self.cal = Platt().fit(self._raw(self.m, Xv), yv)
        return self


class DenseLR(Base):
    """Logistic regression on precomputed dense vectors (e.g. RoBERTa embeddings).
    Texts are row ids into a lookup table supplied at construction."""
    def __init__(self, table):
        self.table = table

    def feat(self, ids):
        return self.table[np.asarray(ids)]

    def _raw(self, m, X):
        return m.decision_function(X)

    def fit(self, tx, y, vx, yv, sample_weight=None):
        X, Xv = self.feat(tx), self.feat(vx)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X, Xv = (X - self.mu) / self.sd, (Xv - self.mu) / self.sd
        fits = [LogisticRegression(C=c / 10, max_iter=3000).fit(X, y) for c in GRID]
        self.m = self._select(fits, Xv, yv)
        self.cal = Platt().fit(self._raw(self.m, Xv), yv)
        return self

    def predict_proba(self, ids):
        X = (self.feat(ids) - self.mu) / self.sd
        return self.cal(self._raw(self.m, X))


def make(name, **kw):
    return {"tfidf_lr": lambda: TfidfLinear("lr", **kw),
            "tfidf_svm": lambda: TfidfLinear("svm", **kw),
            "nbsvm": lambda: NBSVM(),
            "glove_lr": lambda: GloveLR(),
            "style_gb": lambda: StyleGB()}[name]()
