
import html, re, torch, torch.nn as nn, numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset              
from torch.utils.data import DataLoader
from utils_bert import collate_fn, collate_fn_unlab
from functools import partial
from models_bert import BERT_models_baseline, BERT_models_DivDis
from train_bert import train_bert_baseline, train_bert_DivDis
from utils import explain_model

ds = load_dataset("tdavidson/hate_speech_offensive")["train"]
shuffled_ds = ds.shuffle(seed = 42)
def prepare(example):
# (i) rename the target column ------------------------------------------------
    example["labels"] = int(example["class"])       # 0 hate, 1 offensive, 2 neither

    # (ii) light tweet cleaning ---------------------------------------------------
    text = html.unescape(example["tweet"])
    text = re.sub(r"http\S+", "URL", text)          # keep URL token
    text = re.sub(r"@\w+", "@USER", text)           # normalise @mentions
    example["text"] = text

    return example

ds = ds.map(prepare, remove_columns=["count",
                                    "hate_speech_count",
                                    "offensive_language_count",
                                    "neither_count",
                                    "class",
                                    "tweet"])
total = len(ds)
n_labeled   = int(0.4 * total)
n_unlabled   = int(0.8 * total)
train_data_labeled = ds.select(range(0, n_unlabled))
train_data_labeled_DivDis = ds.select(range(0, n_labeled))
train_data_unlabeled = ds.select(range(n_labeled, n_unlabled))
test_data  = ds.select(range(n_unlabled, total))

tok = AutoTokenizer.from_pretrained("bert-base-uncased")


labelled_loader = DataLoader(
    train_data_labeled,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, tokenizer=tok)
)

labelled_loader_DivDis = DataLoader(
    train_data_labeled_DivDis,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, tokenizer=tok)
)
unlabeled_loader = DataLoader(
    train_data_unlabeled,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn_unlab, tokenizer=tok)
)

test_loader  = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    collate_fn=partial(collate_fn, tokenizer=tok)
)

# model = BERT_models_baseline(pretrain_model="bert-base-uncased",
#     mlp_hidden=[64,32],
#     mlp_dropout=0.1,
#     num_classes=3)
   
# best_model, hist = train_bert_baseline(
#     model,
#     train_loader=labelled_loader,
#     val_loader=test_loader,   # or a separate val_loader
#     num_epochs=1
# )

div_model = BERT_models_DivDis(
    pretrain_model="bert-base-uncased",
    num_heads=3,
    num_classes=3,
    mlp_dropout=0.2,
    diversity_weight=1e-3
)

criterion = nn.CrossEntropyLoss()

full_loss  = train_bert_DivDis(
    model = div_model,
    train_loader = labelled_loader_DivDis,
    diverse_loader = unlabeled_loader,
    val_loader = test_loader,
    criterion = criterion,
    num_epochs = 3,
    learning_rate = 2e-5
)
# train_columns = ["hate_speech", "offensive_language", "neither"]
#best_model.eval()
# for head in range(div_model.num_heads):
#     model_name = "DivDis model head_ {}".format(head)
#     model_head = HeadWrapper(div_model, head)
#     model_head.eval()
#     explain_model(model_head, div_model.num_classes, test_data, model_name, input_dim, train_columns)