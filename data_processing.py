import re
import html
import json

def fixup(x):
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace(' @.@ ','.').replace(' @,@ ',',').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(data_name, path):
    if data_name == "aclImdb":
        texts = []
        CLASSES = ['neg', 'pos', 'unsup']
        for idx,label in enumerate(CLASSES):
            for fname in (path/label).glob('*.*'):
                text = fname.open('r', encoding='utf-8').read()
                texts.append(fixup(text))
        return texts

    elif data_name == "RACE":
        texts = set([])
        CLASSES = ['high', 'middle']  # , 'unsup']
        for idx, cla in enumerate(CLASSES):
            for fname in (path / cla).glob('*.*'):
                with open(fname) as dataset_file:
                    data = json.load(dataset_file)
                    texts.add(data['article'])

        return list(texts)
    else:
        raise ValueError("Invalid data name!")
