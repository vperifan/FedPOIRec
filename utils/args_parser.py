import argparse


def str2bool(v):
    return v.lower() in 'true'


def get_config():
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--train_dataset', type=str, default='datasets/athens/athens_checkins_train.txt',
                        help='Train dataset path')
    parser.add_argument('--test_dataset', type=str, default='datasets/athens/athens_checkins_test.txt',
                        help='Test dataset path')

    # train arguments
    parser.add_argument('--mode', type=str, default='federated', help='centralized or federated')
    parser.add_argument('--model', type=str, default='caser', help='The model to user')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

    # federated arguments
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--federated_optimizer', type=str, default='sgd', help="Optimizer to use: adam or sgd")
    parser.add_argument('--fraction', type=float, default=0.1, help='Fraction of users to consider for training')
    parser.add_argument('--log_per', type=int, default=10, help='Verbose details in federated training')
    parser.add_argument('--federated_batch', type=int, default=512, help='Batch size for FL clients')
    parser.add_argument('--federated_evaluate_per', type=int, default=1, help='Evaluate per x epochs')
    parser.add_argument('--aggregation', type=str, default='simple', help='Averaging function: simple or fedavg')
    parser.add_argument('--federated_selection_seed', type=int, default=0, help='Seed for client sampling')

    # arguments for centralized learning
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size to use')
    parser.add_argument('--evaluate_per', type=int, default=1, help='Evaluate per x epochs')

    # common arguments for both centralized and federated learning
    parser.add_argument('--cuda', type=str2bool, default=True, help='Use cuda')
    parser.add_argument('--negatives', type=int, default=3, help='Negative samples')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-6, help='L2 regularization')
    parser.add_argument('--d', type=int, default=128)

    # caser specific arguments
    parser.add_argument('--L', type=int, default=5, help='Embedding Lookup')
    parser.add_argument('--T', type=int, default=3, help='Targets len')
    parser.add_argument('--nv', type=int, default=4, help='Number of vertical layers')
    parser.add_argument('--nh', type=int, default=16, help='Number of horizontal layers')
    parser.add_argument('--drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--ac_conv', type=str, default='relu', help='activation for convolutional layers')
    parser.add_argument('--ac_fc', type=str, default='relu', help='activation for fully connected layers')

    config = parser.parse_args()

    return config
