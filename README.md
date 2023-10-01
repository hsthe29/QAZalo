# Bài toán Question Answering trong cuộc thi Zalo AI Challenge 2019
Thực tập kỹ thuật

**@author: hsthe29**

## Overview 
Given a question, related paragraphs from a Wikipedia article (in the shuffle order), the task is finding paragraph which answers the question of each test case.

F1 measure based on precision and recall is used to rank competing submissions. 

For detailed, visit: https://challenge.zalo.ai/portal/question-answering

## About Data?

Visit: https://challenge.zalo.ai/portal/question-answering

## Approaches 

I used pretrained bert (`bert-base-multilingual-cased`) for fine-tuning

## Hyperparameter
num_train_epochs = 6.0

max_seq_length = 512 

train_batch_size = 16

learning_rate = 2e-5

# Run
Run the project using `flags`. For more details on the flags defined, read the `main.py` file or run this command: `python main.py -h`

### Train 
Please download the init checkpoint at [here]() and put into the folder checkpoint.

Run command ``` sh train.sh ```


### Predict
If you want to only predict, please download the checkpoint of trained model at [here]()

Run command: ``` sh predict.sh ```. 

## Result:
Training logs are store at `logs/`. 

For visualizing: ```$ tensorboard --logdir=logs/tensorboard```

With validation set (`data/train/val.csv`), this model reach 83.6% of F1 score


