# Copyright (C) 2021 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

import os
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast


# wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# tar -xf aclImdb_v1.tar.gz
def get_dataset(data_path):
    def read_imdb_split(split_dir):
        split_dir = Path(split_dir)
        texts = []
        labels = []
        for label_dir in ["pos", "neg"]:
            for text_file in (split_dir / label_dir).iterdir():
                texts.append(text_file.read_text())
                labels.append(0 if label_dir == "neg" else 1)

        return texts, labels

    train_texts, train_labels = read_imdb_split(os.path.join(data_path, "train"))
    test_texts, test_labels = read_imdb_split(os.path.join(data_path, "test"))
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset
