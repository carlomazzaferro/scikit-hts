import re
import os

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

custom_stopwords = {'the', 'to', 'and', 'of', 'in', 'are', 'target', 'business', 'redemption', 'complete', 'initial',
                    'sponsor', 'redeem', 'tender', 'consummate', 'offer', 'proposed', 'table', 'content', 'could',
                    'quarter', 'forward', 'looking', 'http', 'www', 'businesswire', 'com', 'prnewswire', 'gaap', 'inc',
                    'diluted', 'share', 'common', 'stock', 'tax', 'seek', 'ordinary', 'right', 'liability', 'llc',
                    'third', 'party', 'holding', 'company', 'corporation', 'adversely', 'first', 'income', 'net',
                    'nasdaq', 'statement', 'shareholder', 'stockholder', 'march', 'ended', 'end', 'non', 'solution',
                    'ebitda', 'may', 'fund', 'trademark', 'sold', 'offering', 'dow', 'jones', 'barclays' 'barclay',
                    'indxx', 'etf', 'etfs', 'index', 'trust', 'per', 'etns', 'co', 'ltd', 'china', 'prc', 'make', 'mr',
                    'incorporation', 'certificate', 'incorporation', 'consummation', 'in', 'warrant',
                    'combination', 'shall', 'item'
                    }


def read_filter(p):
    for path in os.listdir(p):
        if path.endswith('.txt'):
            with open(os.path.join(p, path), encoding='utf-8') as inf:
                r = inf.read()
                if len(r) == 0:
                    continue
                else:
                    yield r, os.path.splitext(os.path.basename(path))[0]


def sanitize_dict(d):
    d = {k: v for k, v in d.items() if any([isinstance(v, str),
                                            isinstance(v, float),
                                            isinstance(v, int),
                                            isinstance(v, bool),
                                            isinstance(v, tuple)])}
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
    return d


def preprocess_text(doc):
    st = stopwords.words('english')
    text = re.sub('[^a-zA-Z]', ' ', doc)
    text = text.lower()
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.split()
    text = [lem.lemmatize(word) for word in text if word not in st]
    text = [t for t in text if t not in custom_stopwords]
    text = " ".join(text)
    return text
