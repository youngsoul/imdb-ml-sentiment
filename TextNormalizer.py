import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk import wordpunct_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import nltk


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.tokenizer = wordpunct_tokenize
        self.lemmatizer = WordNetLemmatizer()

    def _is_stopword(self, token):
        return token.lower() in self.stopwords

    def _remove_html_lower(self, text):
        text = re.sub('<[^>]*', '', text)

        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

        text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

        return text

    def _lemmatize(self, token, pos_token_tag):
        """
        1.	CC	Coordinating conjunction
	2.	CD	Cardinal number
	3.	DT	Determiner
	4.	EX	Existential there
	5.	FW	Foreign word
	6.	IN	Preposition or subordinating conjunction
	7.	JJ	Adjective
	8.	JJR	Adjective, comparative
	9.	JJS	Adjective, superlative
	10.	LS	List item marker
	11.	MD	Modal
	12.	NN	Noun, singular or mass
	13.	NNS	Noun, plural
	14.	NNP	Proper noun, singular
	15.	NNPS	Proper noun, plural
	16.	PDT	Predeterminer
	17.	POS	Possessive ending
	18.	PRP	Personal pronoun
	19.	PRP$	Possessive pronoun
	20.	RB	Adverb
	21.	RBR	Adverb, comparative
	22.	RBS	Adverb, superlative
	23.	RP	Particle
	24.	SYM	Symbol
	25.	TO	to
	26.	UH	Interjection
	27.	VB	Verb, base form
	28.	VBD	Verb, past tense
	29.	VBG	Verb, gerund or present participle
	30.	VBN	Verb, past participle
	31.	VBP	Verb, non-3rd person singular present
	32.	VBZ	Verb, 3rd person singular present
	33.	WDT	Wh-determiner
	34.	WP	Wh-pronoun
	35.	WP$	Possessive wh-pronoun
	36.	WRB	Wh-adverb
        :param token:
        :param pos_tag:
        :return:
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_token_tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def _normalize(self, document):
        doc2 = self._remove_html_lower(document)
        tokens = self.tokenizer(doc2)
        normalized_tokens = [self._lemmatize(pt[0], pt[1]) for pt in pos_tag([t for t in tokens if not self._is_stopword(t)])]

        return normalized_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            #print(document)
            #yield self._remove_html_lower(document)
            yield ' '.join(self._normalize(document))
