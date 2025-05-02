import torch
def collate_fn(
    batch,
    tokenizer,
    max_len: int = 64
):
    texts, labels = zip(*[(ex["text"], ex["labels"]) for ex in batch])
    
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    return (
        enc["input_ids"], 
        enc["attention_mask"], 
        torch.tensor(labels, dtype=torch.long)
    )

def collate_fn_unlab(
    batch,
    tokenizer,
    max_len: int = 64
):
    """
    Batch ➔ (input_ids, attention_mask)
    Drops any 'labels' in the examples.
    """
    texts = [ex["text"] for ex in batch]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]