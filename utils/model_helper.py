import copy
import time

import numpy as np
import torch

from utils.metrics import evaluate_ranking
from utils.models import Caser, BPR
from utils.train_utils import minibatch, shuffle
from utils.aggregation import avg, fed_avg


class Recommender(object):
    """
    Models are trained by tuples of (users, sequences, targets, negatives)
    Negative sampling: for any known tuple of (user, sequence, targets) one
        or more items are randomly sampled as negatives

    Parameters
    ----------
    epochs: int
        Number of epochs
    batch_size: int
        Batch size
    l2: float
        l2 regularization parameter
    negatives: int
        Number of negative samples to generate for each target
        If targets=3 and negatives=3, then it will sample 9 negatives
    learning_rate: float
        Learning rate
    cuda: bool
        Run the model on GPU or CPU
    model_args: args
        Model arguments, e.g. dimension
    """

    def __init__(self, epochs=None, batch_size=None, l2=None, negatives=None, learning_rate=None, cuda=False,
                 model_args=None):

        # model params
        self.num_items_ = None
        self.num_users_ = None
        self.net_ = None
        self.model_args = model_args
        self.model = model_args.model
        assert self.model in ('caser', 'bpr')

        # learning params
        self.batch_size_ = batch_size
        self.epochs_ = epochs
        self.learning_rate_ = learning_rate
        self.l2_ = l2
        self.negatives_ = negatives
        is_cuda_available = torch.cuda.is_available()
        self.device_ = torch.device("cuda" if cuda and is_cuda_available else "cpu")

        # evaluation
        self.test_sequence = None
        self.candidate_ = dict()

    @property
    def _initilized(self):
        return self.net_ is not None

    def _initialize(self, interactions):
        self.num_items_ = interactions.num_items
        self.num_users_ = interactions.num_users

        if self.model == 'caser':

            self.test_sequence = interactions.test_sequences

            self.net_ = Caser(self.num_users_,
                              self.num_items_,
                              self.model_args).to(self.device_)
        elif self.model == 'bpr':
            self.net_ = BPR(self.num_users_,
                            self.num_items_,
                            self.model_args).to(self.device_)

        self.optimizer_ = torch.optim.Adam(self.net_.parameters(),
                                           weight_decay=self.l2_,
                                           lr=self.learning_rate_)

    def _initialize_global_model(self):
        self.global_model = copy.deepcopy(self.net_)

    def _initialize_local_model(self):
        self.local_model = copy.deepcopy(self.global_model)

    def train(self, train, test, args, verbose=False):
        """
        Training loop

        Parameters
        ----------
        train: training instances, also contains test sequences
        test: targets for test sequences
        args: the train arguments
        verbose: log
        """
        if self.model == 'caser':
            # sequences, targets and users conversion
            sequences_np = train.sequences.sequences
            targets_np = train.sequences.targets
            users_np = train.sequences.user_ids.reshape(-1, 1)

            n_train = sequences_np.shape[0]

        elif self.model == 'bpr':
            user_ids = train.user_ids.astype(np.int64)
            item_ids = train.item_ids.astype(np.int64)

            n_train = train.interactions

        else:
            raise NotImplementedError

        print(f"    Total training instances: {n_train}")

        if not self._initilized:
            self._initialize(train)

        for epoch in range(self.epochs_):
            t1 = time.time()
            self.net_.train()  # set train mode

            if self.model == 'caser':
                users_np, sequences_np, targets_np = shuffle(users_np, sequences_np, targets_np)

                negatives_np = self._generate_negative_samples(users_np, train, n=self.negatives_)

                users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                        torch.from_numpy(sequences_np).long(),
                                                        torch.from_numpy(targets_np).long(),
                                                        torch.from_numpy(negatives_np).long())

                users, sequences, targets, negatives = (users.to(self.device_),
                                                        sequences.to(self.device_),
                                                        targets.to(self.device_),
                                                        negatives.to(self.device_))

                epoch_loss = 0.0

                for (batch_idx, (b_users, b_sequences, b_targets, b_negatives)) in enumerate(minibatch(
                        users, sequences, targets, negatives, batch_size=self.batch_size_)):
                    items_to_predict = torch.cat((b_targets, b_negatives), 1)
                    items_prediction = self.net_(b_sequences, b_users, items_to_predict)

                    (targets_prediction, negatives_prediction) = torch.split(
                        items_prediction, [b_targets.size(1), b_negatives.size(1)], dim=1)

                    self.optimizer_.zero_grad()

                    # BCE loss
                    positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction)))
                    negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction)))
                    loss = positive_loss + negative_loss
                    epoch_loss += loss.item()

                    loss.backward()
                    self.optimizer_.step()

                epoch_loss /= batch_idx + 1

            elif self.model == 'bpr':
                users_np, items_np = shuffle(user_ids, item_ids)

                users, items = (torch.from_numpy(users_np).long(),
                                torch.from_numpy(items_np).long())

                users, items = (users.to(self.device_),
                                items.to(self.device_))

                epoch_loss = 0.0

                for (batch_idx, (b_users, b_items)) in enumerate(
                        minibatch(users, items, batch_size=self.batch_size_)):
                    positive_prediction = self.net_(b_users, b_items)
                    negative_prediction = self._get_negative_prediction(b_users)

                    self.optimizer_.zero_grad()

                    loss = (1.0 - torch.sigmoid(positive_prediction - negative_prediction))
                    loss = loss.mean()
                    epoch_loss += loss.item()

                    loss.backward()
                    self.optimizer_.step()

                epoch_loss /= batch_idx + 1

            t2 = time.time()

            if verbose and (epoch + 1) % args.evaluate_per == 0:
                precision, recall, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])

                print(f"    Epoch {epoch + 1} [{t2 - t1:.2f} s]\n"
                      f"      Loss: {epoch_loss:.4f}, map={mean_aps:.4f}, prec@1={np.mean(precision[0]):.4f}, "
                      f"prec@5={np.mean(precision[1]):.4f}, prec@10={np.mean(precision[2]):.4f}, "
                      f"recall@1={np.mean(recall[0]):.4f}, recall@5={np.mean(recall[1]):.4f}, "
                      f"recall@10={np.mean(recall[2]):.4f}, [{time.time() - t2:.2f} s]")
            else:
                print(f"    Epoch {epoch + 1} [{t2 - t1:.2f} s]\n"
                      f"    Loss: {epoch_loss:.4f}")
            if str(epoch_loss) == "nan":
                print("NaN loss, stopping")
                break

    def train_worker(self, train, test, args, verbose=False):
        """
        Training loop for selected clients in a global round, federated logic
        """

        if self.model == 'caser':
            sequences_np = train.sequences.sequences
            targets_np = train.sequences.targets
            users_np = train.sequences.user_ids.reshape(-1, 1)
            unique_users = np.unique(users_np)

            n_train = sequences_np.shape[0]
        elif self.model == 'bpr':
            user_ids = train.user_ids.astype(np.int64)
            item_ids = train.item_ids.astype(np.int64)

            print(f"    Number of items:  {len(np.unique(item_ids))}")

            unique_users = np.unique(user_ids)
            n_train = train.interactions
        else:
            raise NotImplementedError

        print(f"    Number of users: {len(unique_users)} \n    Number of interactions: {n_train}")

        if not self._initilized:
            self._initialize(train)

        self._initialize_global_model()  # initalize global model
        self.global_model.train()

        users_selected = {}

        batch_size = self.model_args.federated_batch

        for epoch in range(self.epochs_):

            t1 = time.time()

            local_weights, local_losses = {}, []
            print(f"\n | Global Training round: {epoch + 1} |\n")

            # get a fraction of users
            m = max(int(args.fraction * self.num_users_), 1)

            if epoch == 0:
                print(f"    Selecting {m} users")

            np.random.seed(args.federated_selection_seed + epoch)  # seed for random choice
            # random choice without replacement
            idxs_users = np.random.choice(range(self.num_users_), m, replace=False)

            # print(idxs_users)
            local_instances = {}

            for i, idx in enumerate(idxs_users):
                if idx not in users_selected:
                    users_selected[idx] = 0
                users_selected[idx] += 1

                self._initialize_local_model()
                self.local_model.train()  # train mode

                if args.federated_optimizer == "adam":
                    local_optimizer = torch.optim.Adam(self.local_model.parameters(),
                                                       weight_decay=self.l2_,
                                                       lr=self.learning_rate_)
                elif args.federated_optimizer == "sgd":

                    local_optimizer = torch.optim.SGD(self.local_model.parameters(),
                                                      weight_decay=self.l2_,
                                                      lr=self.learning_rate_)

                if self.model == 'caser':

                    user_indices = np.where(users_np == idx)[0]
                    # get the user and its corresponding indices and targets
                    user_np = np.array([[users_np[i][0]] for i in user_indices])
                    sequence_np = np.array([list(sequences_np[i]) for i in user_indices])
                    target_np = np.array([list(targets_np[i]) for i in user_indices])

                    n_train = len(user_np)
                    local_instances[idx] = n_train
                    negative_np = self._generate_negative_samples(user_np, train, n=self.negatives_)

                    if i % args.log_per == 0:
                        print(f"    Training instances for user {idx}: {n_train}")
                        # print(f"    Negative samples: {len(negative_np)}")

                elif self.model == 'bpr':
                    user_indices = np.where(user_ids == idx)[0]

                    # get the user and its corresponding indices and targets
                    user_np = np.array([[user_ids[i]] for i in user_indices])
                    item_np = np.array([list(item_ids)[i] for i in user_indices])

                    n_train = len(user_np)
                    local_instances[idx] = n_train

                    if i % args.log_per == 0:
                        print(f"    Training instances for user {idx}: {n_train}")

                for local_epoch in range(self.model_args.local_epochs):

                    epoch_loss = 0.0
                    if self.model == 'caser':
                        user_np, sequence_np, target_np = shuffle(user_np, sequence_np, target_np)
                        user, sequence, target, negative = (torch.from_numpy(user_np).long(),
                                                            torch.from_numpy(sequence_np).long(),
                                                            torch.from_numpy(target_np).long(),
                                                            torch.from_numpy(negative_np).long())

                        user, sequence, target, negative = (user.to(self.device_),
                                                            sequence.to(self.device_),
                                                            target.to(self.device_),
                                                            negative.to(self.device_))
                        for (batch_idx, (b_user, b_sequence, b_target, b_negative)) in enumerate(minibatch(
                                user, sequence, target, negative, batch_size=batch_size)):
                            item_to_predict = torch.cat((b_target, b_negative), 1)
                            item_prediction = self.local_model(b_sequence, b_user, item_to_predict)

                            if len(item_prediction.shape) == 1:
                                (target_prediction, negative_prediction) = torch.split(
                                    item_prediction, [b_target.size(1), b_negative.size(1)])
                            else:

                                (target_prediction, negative_prediction) = torch.split(
                                    item_prediction, [b_target.size(1), b_negative.size(1)], dim=1)

                            local_optimizer.zero_grad()

                            pos_loss = -torch.mean(torch.log(torch.sigmoid(target_prediction)))
                            neg_loss = -torch.mean(torch.log(1 - torch.sigmoid(negative_prediction)))
                            local_loss = pos_loss + neg_loss
                            epoch_loss += local_loss.item()

                            local_loss.backward()

                            local_optimizer.step()

                        epoch_loss /= batch_idx + 1
                        if local_epoch == 0:
                            tmp_loss = epoch_loss

                    elif self.model == 'bpr':
                        user_np, item_np = shuffle(user_np, item_np)
                        user, item = (torch.from_numpy(user_np).long(),
                                      torch.from_numpy(item_np).long())

                        user, item = (user.to(self.device_),
                                      item.to(self.device_))
                        for (batch_idx, (b_user, b_item)) in enumerate(
                                minibatch(user, item, batch_size=batch_size)):
                            positive_prediction = self.local_model(b_user, b_item)
                            negative_prediction = self._get_negative_prediction(b_user, local_model=self.local_model)
                            local_optimizer.zero_grad()

                            local_loss = (1.0 - torch.sigmoid(positive_prediction - negative_prediction))
                            local_loss = local_loss.mean()
                            epoch_loss += local_loss.item()

                            local_loss.backward()

                            local_optimizer.step()

                        epoch_loss /= batch_idx + 1
                        if local_epoch == 0:
                            tmp_loss = epoch_loss
                if i % args.log_per == 0:
                    print(
                        f"        Initial Loss: {tmp_loss:.4f} -- Loss after {args.local_epochs} "
                        f"epochs: {epoch_loss:.4f}")

                local_losses.append(epoch_loss)

                local_weights[idx] = copy.deepcopy(self.local_model.state_dict())

            t2 = time.time()  # end timer

            if args.aggregation == "fedavg":
                aggregated_global_weights = fed_avg(self.global_model.state_dict(), local_weights, local_instances)
            elif args.aggregation == "simple":
                aggregated_global_weights = avg(self.global_model.state_dict(), local_weights, local_instances)
            else:
                raise NotImplementedError
            self.global_model.load_state_dict(aggregated_global_weights)

            self.net_.load_state_dict(aggregated_global_weights)

            # evaluate global model
            if verbose and ((epoch + 1) % args.federated_evaluate_per) == 0:
                precision, recall, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])
                print(f"    [{t2 - t1:.2f} s]\n"
                      f"    Average Loss: {np.mean(local_losses):.4f}, map={mean_aps:.4f}, prec@1={np.mean(precision[0]):.4f}, "
                      f"prec@5={np.mean(precision[1]):.4f}, prec@10={np.mean(precision[2]):.4f}, "
                      f"recall@1={np.mean(recall[0]):.4f}, recall@5={np.mean(recall[1]):.4f}, "
                      f"recall@10={np.mean(recall[2]):.4f}, [{time.time() - t2:.2f} s]")

            if str(np.mean(local_losses)) == "nan":
                print("NaN Loss, stopping")
                break

    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negative items for each user. The candidate negatives are {All items} \ {Items Interacted}
        Parameters
        ---------
        users: user ids
        interactions: training instances
        total number of negative items to sample for each sequence
        """
        if len(users) == 1:
            users_ = np.array([users.squeeze()])
        else:
            users_ = users.squeeze()

        negative_samples = np.zeros((users_.shape[0], n), np.int64)

        if not self.candidate_:
            all_items = np.arange(interactions.num_items - 1) + 1
            train = interactions.tocsr()

            for user, row in enumerate(train):
                self.candidate_[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self.candidate_[u]
                negative_samples[i, j] = x[np.random.randint(len(x))]

        return negative_samples

    def _get_negative_prediction(self, user_ids, local_model=None):
        negative_items = self._sample_items(self.num_items_, len(user_ids))

        negative_var = torch.from_numpy(negative_items).long()
        negative_var = negative_var.to(self.device_)

        if local_model is None:
            negative_prediction = self.net_(user_ids, negative_var)
        else:
            negative_prediction = local_model(user_ids, negative_var)

        return negative_prediction

    def _sample_items(self, num_items, shape):
        """
        Randomly sample a number of items
        :param num_items: int, the total number of items for which we should sample:
                            the max_value if a sampled item id will be smaller than this parameter
        :param shape: int or tuple of ints
                    Shape of the sampled array
        :return: sampled item ids
        """
        items = np.random.randint(0, num_items, shape, dtype=np.int64)
        return items

    def predict(self, user_id, item_ids=None):
        """
        Predict for a user.
        Caser Model:
            Retrieves the test sequence associated with the user_id and
            compute recommendation scores for items
        BPR Model:
            Given a user id, compute the recommendation scores for items

        Parameters
        ----------
        user_id: user id -> Int or array
                        If int, it will predict the recommendation scores for this user
                        for all items in item_ids.
                        If array, it will predict scores for all (user, item) pairs
                        defined by user_ids and item_ids
        item_ids: array containing the item ids for which prediction scores are desired. If not supplied,
                                                                                predictions for all items are computed.
        """
        if self.model == 'caser':
            if self.test_sequence is None:
                raise ValueError('Missing test sequences, cannot make predictions!')

            self.net_.eval()  # evaluation mode
            with torch.no_grad():
                sequences_np = self.test_sequence.sequences[user_id, :]
                sequences_np = np.atleast_2d(sequences_np)

                if item_ids is None:
                    item_ids = np.arange(self.num_items_).reshape(-1, 1)

                sequences = torch.from_numpy(sequences_np).long()
                item_ids = torch.from_numpy(item_ids).long()
                user_id = torch.from_numpy(np.array([[user_id]])).long()

                user, sequences, items = (
                    user_id.to(self.device_), sequences.to(self.device_), item_ids.to(self.device_))

                output = self.net_(sequences, user, items, for_pred=True)

                return output.cpu().numpy().flatten()
        elif self.model == 'bpr':
            self.net_.eval()  # evaluation mode
            with torch.no_grad():
                if item_ids is None:
                    item_ids = np.arange(self.num_items_, dtype=np.int64)
                if np.isscalar(user_id):
                    user_id = np.array(user_id, dtype=np.int64)
                user_id = torch.from_numpy(user_id.reshape(-1, 1).astype(np.int64))
                item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

                if item_ids.size()[0] != user_id.size(0):
                    user_id = user_id.expand(item_ids.size())

                user_var = user_id.to(self.device_).squeeze()
                item_var = item_ids.to(self.device_).squeeze()

                output = self.net_(user_var, item_var)
                return output.cpu().numpy().flatten()
