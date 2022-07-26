def prepare_batch(batch):
    import jax.numpy as jnp
    for key in batch.keys():
        batch[key] = jnp.array(batch[key])

def load_wikitext2(batch_size=100, max_len=128, data_dir="wikitext"):
    """Download (if needed) and load WikiText-2 dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding
    import numpy as np
    import jax.numpy as jnp
    # import pdb; pdb.set_trace()
    raw_dataset = load_dataset(data_dir, "wikitext-2-raw-v1", split="train")

    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_dataset = raw_dataset.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported
        # it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_len) * max_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_len] for i in range(0, total_length, max_len)]
            for k, t in concatenated_examples.items()
        }
        # result["input_ids"] = torch.LongTensor(result["input_ids"])
        # result["labels"] = result["input_ids"].clone().numpy()
        # if result["input_ids"].shape[0] == 0:
        #     result["input_ids"] = result["input_ids"].numpy()
        #     return result

        # # create random array of floats in equal dimension to input_ids
        # rand = torch.rand(result["input_ids"].shape)
        # # where the random array is less than 0.15, we set true
        # mask_arr = (rand < 0.15) * (result["input_ids"] != 101) * (result["input_ids"] != 102)
        # # create selection from mask_arr
        # selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
        # # apply selection index to inputs.input_ids, adding MASK tokens
        # result["input_ids"][:, selection] = 103

        # result["input_ids"] = result["input_ids"].numpy()
        result["position_ids"] = np.ones(np.array(result["input_ids"]).shape, dtype=np.int32)

        result["labels"] = np.copy(result["input_ids"])

        return result

    tokenized_dataset = tokenized_dataset.map(
        group_texts, 
        batched=True, 
        batch_size=batch_size, 
        num_proc=4)
    ds = tokenized_dataset.with_format(type='numpy')
    ds = ds.to_dict(batch_size, batched=True)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
    # params = data_collator(tokenized_dataset[:batch_size])
    # print("HERE", type(params))
    # return params.to_tf_dataset()
    return ds
