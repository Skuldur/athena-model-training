import glob
import os 
import inspect
import importlib

dir_path = os.path.dirname(os.path.realpath(__file__))

base_intents = ['BaseIntent', 'NeuralIntent', 'IntentClassifier']

def get_all_intents():
    files = os.listdir(path=dir_path)

    intent_files = list(filter(lambda x: 'intent' in x, files))

    intent_map = {}

    # Loop through all intent files
    for file in intent_files:
        key = file.split('.py')[0]
        i = importlib.import_module('chalicelib.intents.en.controllers.%s' % key)
        # go through all members in the module

        for name, obj in inspect.getmembers(i):
            # filter down to classes only
            if inspect.isclass(obj):

                # TODO: Add better filter
                # This filter is vulnerable to duplicates if we start doing
                # class inheritance
                if 'Intent' in obj.__name__ and not obj.__name__ in base_intents:
                    intent_map[key] = obj()

    return intent_map

def get_intent(intent_name):
    i = importlib.import_module('chalicelib.intents.en.controllers.%s' % intent_name)

    intent = None

    for name, obj in inspect.getmembers(i):
            # filter down to classes only
            if inspect.isclass(obj):

                # TODO: Add better filter
                # This filter is vulnerable to duplicates if we start doing
                # class inheritance
                if 'Intent' in obj.__name__ and not obj.__name__ in base_intents:
                    print(obj.__name__)
                    intent = obj()

    return intent



