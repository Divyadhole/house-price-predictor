# Architecture

`CSV -> schema validation -> ratio features -> seeded split -> train-only ColumnTransformer -> Ridge and histogram-gradient candidates -> held-out selection -> joblib artifact -> CLI/API`.

The API never fits preprocessors or models. It loads the persisted artifact and sends the typed request through the same feature function and selected pipeline used during evaluation.
