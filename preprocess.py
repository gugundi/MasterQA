import os
import sys
import json
import pickle

import nltk
import tqdm
from PIL import Image

image_index = {'gqa': 'imageId'}


def process_question(root, split, word_dic=None, answer_dic=None, dataset_type='gqa'):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions', f'{split}_balanced_questions.json')) as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data.values()):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']

        try:
            answer = answer_dic[answer_word]
        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question[image_index[dataset_type]], question_token, answer))

    with open(f'dataset/{dataset_type}_{split}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


if __name__ == '__main__':
    dataset_type = sys.argv[1]
    root = sys.argv[2]
    word_dic, answer_dic = process_question(root, 'train', dataset_type='gqa')
    process_question(root, 'val', word_dic, answer_dic, dataset_type='gqa')

    with open(f'dataset/{dataset_type}_dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
