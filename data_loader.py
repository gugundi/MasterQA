import numpy as np

from torch.utils.data import Dataset

class GQA(Dataset):
    def __init__(self, root, split='train', transform=None):
        with open(f'dataset/gqa_{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)

        self.root = root
        self.split = split

    def __getitem__(self, index):
        imgFile, question, answer = self.data[index]

        return imgFile, question, len(question), answer

    def __len__(self):
        return len(self.data)
'''
Load a batch

def collate_data(batch):
    images, lengths, answers = [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers)

'''
