# FedPOIRec

Implementation of the paper:

*FedPOIRec: Privacy Preserving Federated POI Recommendation with Social
  Influence, Vasileios Perifanis, George Drosatos, Giorgos Stamatelatos and Pavlos
  S. Efraimidis, 2021*

# Requirements
* Python 3
* PyTorch 
* NumPy
* SciPy

## Installation
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install scipy
```

# Dataset
The datasets are splitted into train and test sets. Each dataset contains triplets of interactions in the form of:
* [UserId, ItemId, Preference],

where Preference is 1 for an interacted item.

# Usage
```
usage: main.py [-h] [--train_dataset TRAIN_DATASET] [--test_dataset TEST_DATASET] 
               [--mode MODE] [--model MODEL] [--epochs EPOCHS] [--local_epochs LOCAL_EPOCHS] 
               [--federated_optimizer FEDERATED_OPTIMIZER] [--fraction FRACTION] [--log_per LOG_PER] 
               [--federated_batch FEDERATED_BATCH] [--federated_evaluate_per FEDERATED_EVALUATE_PER] 
               [--aggregation AGGREGATION] [--federated_selection_seed FEDERATED_SELECTION_SEED]
               [--seed SEED] [--batch_size BATCH_SIZE] [--evaluate_per EVALUATE_PER]
               [--cuda CUDA] [--negatives NEGATIVES] [--learning_rate LEARNING_RATE]
               [--l2 L2] [--d D] 
```
* train_dataset/test_dataset: train and test datasets
* mode: centralized or federated
* model: caser or bpr
* epochs: epochs for the centralized training or global rounds for the federated setting
* local_epochs: local epochs for the federated setting
* federated_optimizer: sgd or adam
* fraction: fraction of users to consider for averaging in the federated setting
* log_per: log interval
* aggregation: simple averaging or FedAvg
* d: the size of the latent dimension

## Usage Examples
```
python main.py --train_dataset datasets/athens/athens_checkins_train.txt --test_dataset datasets/athens/athens_checkins_test.txt --mode centralized --model bpr --epochs 150 --cuda True --learning_rate 1e-3 --l2 1e-6 --d 128

python main.py --train_dataset datasets/athens/athens_checkins_train.txt --test_dataset datasets/athens/athens_checkins_test.txt --mode federated --model bpr --epochs 150 --cuda True --federated_optimizer sgd --learning_rate 1e-1 --l2 1e-6 --d 128 --aggregation simple --log_per 100

python main.py --train_dataset datasets/athens/athens_checkins_train.txt --test_dataset datasets/athens/athens_checkins_test.txt --mode centralized --model caser --epochs 50 --cuda True --learning_rate 1e-3 --l2 1e-6 --d 128

python main.py --train_dataset datasets/athens/athens_checkins_train.txt --test_dataset datasets/athens/athens_checkins_test.txt --mode federated --model caser --epochs 150 --cuda True --federated_optimizer sgd --learning_rate 1e-1 --l2 1e-6 --d 128 --aggregation simple --log_per 100
```
