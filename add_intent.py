import argparse
import os
import json
from distutils.dir_util import copy_tree
from intent_train import train
from intent_classifier import train as train_classifier

parser = argparse.ArgumentParser(description='Train a new intent')
parser.add_argument('--schema_file', '-s', type=str,
                    help='The path to the intent schema')


def run(schema_path):
    with open(schema_path, encoding='utf-8') as schema_text:
        schema = json.load(schema_text)

        root_dir = os.path.join(create_folder_structure(schema.get('name')), 'chalicelib')

        lang = schema.get("language")
        if lang:
            train(schema, lang=lang, root_dir=root_dir)
        else:
            train(schema, root_dir=root_dir)

        if lang:
            train_classifier(lang)
        else:
            train_classifier()

def create_folder_structure(name):
    if not name:
        raise Exception('Schema requires a name field')

    lambda_name = '%s-lambda' % name

    # Only create it if it doesn't already exist
    if not os.path.exists(lambda_name):
        os.mkdir(lambda_name)
        copy_tree('./lambda_template/', lambda_name)

    return lambda_name


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.schema_file)