import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from datasets import Dataset
class CoreJurCorpus(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        batch_size: int = 2,
        tokenizer = None,
        max_len = 256
    ):
        super().__init__()

        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def get_label_list(self, labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples['tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                else:
                    label_ids.append(self.labels_to_ids[label[word_idx]])                               

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        #tokenized_inputs = {key: torch.as_tensor(val) for key, val in tokenized_inputs.items()}
        return tokenized_inputs

    def setup(self, stage=None):
        
        raw_datasets = {}
        raw_datasets['train'] = Dataset.from_json(self.train_path)
        raw_datasets['test'] = Dataset.from_json(self.test_path)
        
        

        label_list = self.get_label_list(raw_datasets["train"]['tags'])
        self.num_labels = len(label_list)

        self.labels_to_ids = {v:k for k,v in enumerate(label_list)}
        self.ids_to_labels = {k:v for k,v in enumerate(label_list)}


        # TOKENIZE TEXT
        train_dataset = raw_datasets['train'].map(
                self.tokenize_and_align_labels,
                batched=True,
                num_proc=2,
                desc="Running tokenizer on train dataset",
            )
        test_dataset = raw_datasets['test'].map(
                self.tokenize_and_align_labels,
                batched=True,
                num_proc=2,
                desc="Running tokenizer on test dataset",
            )
        self.train_dataset = train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'tags'])
        self.test_dataset = test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'tags'])

 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=2)