# CSCI-544-HW-3

Bidirectional LSTM NER with and without GloVe

Environment
Python 3.12

PyTorch (tested with 2.x)

NumPy

Files
dataset.py

Data loading utilities:

read_conll(path, has_tags)

build_word_vocab, build_tag_vocab

NERDataset, NERTestDataset

collate_train_with_vocab, collate_test_with_vocab

models.py

BiLSTMTagger: BLSTM with learned embeddings (Task 1).

BiLSTMTaggerWithEmb: BLSTM with GloVe‑initialized embeddings (Task 2).

glove_utils.py

load_glove(path, emb_dim)

build_embedding_matrix(word2idx, glove_dict, emb_dim)

train.py

Trains Task 1 or Task 2 models and saves:

blstm1.pt / blstm2.pt

dev1.out / dev2.out (predictions on dev)

predict.py

Loads a saved model and generates:

test1.out / test2.out (predictions on test)

eval/eval.py, eval/conll03eval

Official CoNLL evaluation script (provided).

How to Train
Run all commands from inside the HW3 directory.

Task 1: Simple BiLSTM (random embeddings)
bash
python train.py --task 1 --epochs 10 --batch_size 32 --lr 0.1

This will: 

Train the BiLSTM model on data/train.

Evaluate on data/dev each epoch.

Save the best model as blstm1.pt.

Write dev predictions as dev1.out.

Task 2: BiLSTM with GloVe

Ensure the GloVe file glove.6B.100d (100‑dimensional vectors, plain text) is in the HW3 directory.

bash
python train.py --task 2 --epochs 10 --batch_size 32 --lr 0.05 --glove_path glove.6B.100d
This will:

Initialize embeddings from GloVe (using case‑sensitive strategy described in the report).

Train the BLSTM model on data/train.

Save the best model as blstm2.pt.

Write dev predictions as dev2.out.

How to Generate Prediction Files
After training is complete and blstm1.pt / blstm2.pt are saved:

Dev predictions (reproducible)
These are generated automatically at the end of training:

Task 1 dev predictions: dev1.out

Task 2 dev predictions: dev2.out

You can regenerate them by rerunning train.py with the same commands as above.

Test predictions
bash
# Task 1 test predictions -> test1.out
python predict.py --task 1 --test_path data/test --model_path blstm1.pt

# Task 2 test predictions -> test2.out
python predict.py --task 2 --test_path data/test --model_path blstm2.pt
Each .out file is in the required CoNLL format:

text
idx word PRED_TAG

How to Evaluate on Dev
Use the official script from inside the eval directory:

bash
cd eval

# Task 1
python eval.py -p ../dev1.out -g ../data/dev

# Task 2
python eval.py -p ../dev2.out -g ../data/dev
This prints precision, recall, and F1 and was used to obtain the numbers as in the report.
