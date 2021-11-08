local bert_model = "bert-base-cased";

{
    "dataset_reader" : {
        "type": "example_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": 128
        },
        "text_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": bert_model
            },
        },
        "to_index": 6,

    },
    "train_data_path": "data/evp_train.jsonl",
    "validation_data_path": "data/evp_dev.jsonl",
    "model": {
        "type": "sentence_level_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model,
                    "train_parameters": true
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": bert_model
        },
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true,
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.00003,
        },
        "num_epochs": 3,
        "cuda_device": 1,
    }
}