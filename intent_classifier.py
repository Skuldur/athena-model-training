from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize as tokenize
import numpy
import json
import os
from os import path
from pathlib import Path
import glob
from sequences import TrainSequence
from preprocessing import IndexTransformer
from model import TextClassification

yes_or_no_training = {
    'en': {
        'yes': [
            ['yes'],
            ['yes', 'please'],
            ['yeah'],
            ['yep'],
            ['sure'],
            ['yuppers'],
            ['right', 'on'],
            ['sure', 'thing'],
            ['uh-huh'],
            ['by', 'all', 'means'],
            ['ok'],
            ['okay'],
            ['all', 'right'],
            ['alright'],
            ['of', 'course']
        ],
        'no': [
            ['no'],
            ['nix'],
            ['nope'],
            ['nah'],
            ['nope'],
            ['nay'],
            ['not', 'now'],
            ['no', 'thanks'],        
            ['maybe', 'another', 'time'],        
        ]
    }
}

training_samples = {
    'en': {
        'pause_intent': [
            ['stop'],
            ['shut', 'up'],
            ['pause']
        ],
        'next_intent': [
            ['next'],
            ['skip'],
            ['skip', 'this', 'song']
        ],
        'resume_intent': [
            ['resume']
        ],
        'read_shopping_intent': [
            ['read', 'my', 'shopping', 'list'],
            ["what's", "on", "my", "shopping", "list"],
            ["what", "is", "on", "my", "shopping", "list"],
            ["what", "do", "i", "have", "on", "my", "shopping", "list"],
            ["what", "do", "i", "need"],
            ["what", "do", "i", "need", "to", "buy"]
        ],
        'party_intent': [
            ["let's", 'get', 'this', 'party', 'started'],
            ["let", "us", 'get', 'this', 'party', 'started'],
            ["put", "on", 'my', 'jam'],
        ]
    },
    'is': {
        'pause_intent': [
            ['stopp'],
            ['þegiðu'],
            ['pásaðu'],
            ['pása']
        ],
        'next_intent': [
            ['næsta', 'lag'],
            ['næsta'],
            ['slepptu', 'þessu', 'lagi']
        ],
        'resume_intent': [
            ['byrjaðu', 'aftur'],
        ],
        'read_shopping_intent': [
            ['lestu', 'innkaupalistann', 'minn'],
            ["hvað", "er", "á", "innkaupalistanum", "mínum"],
            ["hvað", "vantar", "mig"]
        ],
        'party_intent': [
            ["tími", "til", "að" "djamma"],
            ["partý", "tími"],
            ["djömmum"]
        ]
    }
}

vocab = set()
vocab.add("<unk>")

TARGET_SIZE = 100

def create_dataset(sample_utter):
    global vocab
    X = []
    Y = []
    intent_labels = []
    maxlen = 300

    # find longest sample length
    for intent_id in sample_utter:
        if maxlen < len(sample_utter[intent_id]):
            maxlen = len(sample_utter[intent_id])

    # get all intent labels
    for intent_id in sample_utter:
        intent_labels.append(intent_id)

    # create dataset
    # go maxlen times through each dataset
    for i in range(0, maxlen):
        for intent_id in sample_utter:
            sample = sample_utter[intent_id]

            for word in sample[i % len(sample)]:
                vocab.add(word)

            X.append(sample[i % len(sample)])
            Y.append([intent_id])

    print(X)

    return X, Y, intent_labels

DATA_PATH = path.join(path.dirname(path.realpath(__file__)), 'data')

def get_intent_data(lang):
    global training_samples

    dataset_path = path.join(DATA_PATH, lang, '*.json')
    for file in glob.glob(dataset_path):
        key = Path(file).stem
        with open(file, encoding='utf-8') as f:
            print(file)
            training_samples[lang][key] = json.load(f)

def get_word_id(vocab, word):
    if word in vocab:
        return vocab[word]
    return vocab['<unk>']


def train_yes_or_no(lang='en'):
    global yes_or_no_training

    X, Y, labels = create_dataset(yes_or_no_training[lang])

    with open('intents/%s/config/labels/yes_or_no_labels.json' % lang, 'w') as fp:
        json.dump(labels, fp)

    x_train, x_test, x_val = numpy.split(X, [int(len(X)*0.75), int(len(X)*0.95)])
    y_train, y_test, y_val = numpy.split(Y, [int(len(X)*0.75), int(len(X)*0.95)])

    vectors = sorted(list(vocab))

    vocab_map = dict((word, number) for number, word in enumerate(vocab))

    with open('intents/%s/config/vocab/yes_or_no_word_vocab.json' % lang, 'w') as fp:
            json.dump(vocab_map, fp)

    preprocessor = IndexTransformer()
    preprocessor.fit(vocab_map, labels)
    n_words = len(vectors)

    train_seq = TrainSequence(x_train, y_train, 32, preprocessor.transform)
    test_seq = TrainSequence(x_test, y_test, 32, preprocessor.transform)
    model = TextClassification(labels, n_words)
    model.build()
    model.compile()
    model.train(train_seq, test_seq)
    weights_path = path.join(path.dirname(path.realpath(__file__)), "intents", lang, "config", "weights", 'yes_or_no-weights.hdf5')
    model.model.save_weights(weights_path)

    sentences = x_val

    idx2label = dict((number, label) for number, label in enumerate(labels))

    wrong = 0

    for sentence, true_label in zip(sentences, y_val):
        words = [w for w in sentence]
        word_id_array = [[get_word_id(vocab_map, w) for w in sentence]]
        p = model.predict(numpy.array(word_id_array))

        if idx2label[p[0]] != true_label:
                wrong += 1

    percentage = 100*(1.0*(len(sentences)-wrong) / len(sentences))
    print("Validation accuracy is %s percent" % percentage)

def train(lang='en'):
    global training_samples

    get_intent_data(lang)

    X, Y, labels = create_dataset(training_samples[lang])

    with open('intents/%s/config/labels/text_classification_labels.json' % lang, 'w') as fp:
        json.dump(labels, fp)

    x_train, x_test, x_val = numpy.split(X, [int(len(X)*0.75), int(len(X)*0.95)])
    y_train, y_test, y_val = numpy.split(Y, [int(len(X)*0.75), int(len(X)*0.95)])

    vectors = sorted(list(vocab))

    vocab_map = dict((word, number) for number, word in enumerate(vocab))

    with open('intents/%s/config/vocab/text_classification_word_vocab.json' % lang, 'w') as fp:
            json.dump(vocab_map, fp)

    preprocessor = IndexTransformer()
    preprocessor.fit(vocab_map, labels)
    n_words = len(vectors)

    train_seq = TrainSequence(x_train, y_train, 32, preprocessor.transform)
    test_seq = TrainSequence(x_test, y_test, 32, preprocessor.transform)
    model = TextClassification(labels, n_words)
    model.build()
    model.compile()
    model.train(train_seq, test_seq)
    weights_path = path.join(path.dirname(path.realpath(__file__)), "intents", lang, "config", "weights", 'text_classification-weights.hdf5')
    model.model.save_weights(weights_path)

    sentences = x_val

    idx2label = dict((number, label) for number, label in enumerate(labels))

    wrong = 0

    for sentence, true_label in zip(sentences, y_val):
        words = [w for w in sentence]
        word_id_array = [[get_word_id(vocab_map, w) for w in sentence]]
        p = model.predict(numpy.array(word_id_array))

        if idx2label[p[0]] != true_label:
                wrong += 1

    percentage = 100*(1.0*(len(sentences)-wrong) / len(sentences))
    print("Validation accuracy is %s percent" % percentage)

class IntentClassifier():
    def __init__(self, type, lang='en'):
        label_path = path.join(path.dirname(path.realpath(__file__)), "intents", lang, "config", "labels", "%s_labels.json" % type)
        with open(label_path) as f:
            intents = json.load(f) 

        # load words
        word_vocab_path = path.join(path.dirname(path.realpath(__file__)), "intents", lang, "config", "vocab", "%s_word_vocab.json" % type)
        with open(word_vocab_path) as f:
            word_vocab = json.load(f)

        n_words = len(word_vocab)
        self.vocab_map = word_vocab
        self.model = TextClassification(intents, n_words)
        self.model.build()

        weights_path = path.join(path.dirname(path.realpath(__file__)), "intents", lang, "config", "weights", '%s-weights.hdf5' % type)
        self.model.load_weights(weights_path)

        self.idx2label = dict((number, label) for number, label in enumerate(intents))


    def predict(self, command):
        sentence =  tokenize(command)

        words = [w.lower() for w in sentence]
        word_id_array = [[get_word_id(self.vocab_map, w) for w in sentence]]
        p = self.model.predict(numpy.array(word_id_array))
        print(self.idx2label[p[0]])
        return self.idx2label[p[0]]


if __name__ == '__main__':
    train(lang='en')