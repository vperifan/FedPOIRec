from utils.args_parser import get_config
from utils.interactions import Interactions
from utils.model_helper import Recommender
from utils.train_utils import set_seed

if __name__ == "__main__":
    config = get_config()
    print(vars(config))

    set_seed(seed=config.seed, cuda=config.cuda)

    if config.mode == "centralized":

        if config.model == "caser":
            # load dataset
            train = Interactions(config.train_dataset)
            # transform triplets to sequences
            train.to_sequence(config.L, config.T)
            test = Interactions(config.test_dataset, user_map=train.user_map, item_map=train.item_map)
        elif config.model == "bpr":
            train = Interactions(config.train_dataset)
            test = Interactions(config.test_dataset, user_map=train.user_map, item_map=train.item_map)

        model = Recommender(epochs=config.epochs, batch_size=config.batch_size, learning_rate=config.learning_rate,
                            l2=config.l2, negatives=config.negatives, model_args=config, cuda=config.cuda)

        model.train(train, test, args=config, verbose=True)

    if config.mode == "federated":

        if config.model == "caser":
            train = Interactions(config.train_dataset)
            train.to_sequence(config.L, config.T)
            test = Interactions(config.test_dataset, user_map=train.user_map, item_map=train.item_map)
        elif config.model == "bpr":
            train = Interactions(config.train_dataset)
            test = Interactions(config.test_dataset, user_map=train.user_map, item_map=train.item_map)

        global_model = Recommender(epochs=config.epochs, batch_size=config.batch_size,
                                   learning_rate=config.learning_rate,
                                   l2=config.l2, negatives=config.negatives, model_args=config, cuda=config.cuda)
        global_model.train_worker(train, test, args=config, verbose=True)
