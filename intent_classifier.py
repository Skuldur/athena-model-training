import json
from os import path
from pathlib import Path
import glob

from intent_trainer.datautils import Vocabulary
from intent_trainer.trainer import Trainer
from intent_trainer.preprocessing import Preprocessor
from intent_trainer.models import TextClassification

training_samples = {
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

def create_dataset(sample_utter):
    X = []
    Y = []
    maxlen = 300

    # find longest sample length
    for intent_id in sample_utter:
        if maxlen < len(sample_utter[intent_id]):
            maxlen = len(sample_utter[intent_id])

    # create dataset
    # go maxlen times through each dataset
    for i in range(0, maxlen):
        for intent_id in sample_utter:
            sample = sample_utter[intent_id]

            X.append(sample[i % len(sample)])
            Y.append([intent_id])

    return X, Y

def get_intent_data(lang):
    DATA_PATH = path.join(path.dirname(path.realpath(__file__)), 'data')
    training_samples = {}

    dataset_path = path.join(DATA_PATH, lang, '*.json')
    for file in glob.glob(dataset_path):
        key = Path(file).stem
        with open(file, encoding='utf-8') as f:
            training_samples[key] = json.load(f)

    return training_samples

def train(lang='en'):
    config_path = path.join(path.dirname(path.realpath(__file__)), "intents", lang, "config")
    training_samples = get_intent_data(lang)
    X, Y = create_dataset(training_samples)

    # get all intent labels
    labels = Vocabulary(UnknownToken='other')
    for intent_id in training_samples:
        labels.add(intent_id)

    word_vocab = Vocabulary()
    word_vocab.add('<unk>')
    word_vocab.build_vocab([w for command in X for w in command])

    batch_size = 32
    epochs = 30
    preprocessor = Preprocessor(WordVocab=word_vocab, LabelVocab=labels)
    model = TextClassification(labels, len(word_vocab))
    trainer = Trainer(model, X, Y, preprocessor, split=[0.75, 0.95])
    trainer.train(batch_size, epochs)
    trainer.evaluate(labels.get_inverse_map())

    model.save_weights(path.join(config_path, 'weights', 'text_classification.hdf5'))
    word_vocab.save(path.join(config_path, "vocab", 'text_classification_word_vocab.json'))
    labels.save(path.join(config_path, "labels", 'text_classification_labels.json'))


if __name__ == '__main__':
    train(lang='is')