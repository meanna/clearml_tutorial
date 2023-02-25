# python text_classification.py --epochs 20 --lr 0.005
import argparse
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from clearml import Task, TaskTypes
from clearml import Dataset

# ######################## ClearML #########################

Task.add_requirements('requirements.txt')

task = Task.init(
    project_name='Text Classification',
    task_name='sentiment analysis',
    task_type=TaskTypes.training,
    tags="my model",
    auto_connect_arg_parser=True,
)

task.execute_remotely(queue_name="<=12GB", clone=False, exit_process=True)

ds = Dataset.get(dataset_id="a74d6f9397674d11949dc97d36e377fb")
ds.list_files()
ds.get_mutable_local_copy(target_folder='./data')

logger = task.get_logger()

# ######################### MODEL #########################

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            label, *tokens = line.split()
            data.append((label, tokens))
    return data


def collate(batch):
    '''
    A custom collate function. Return 3 tensors: word-IDs, labels, text length
    '''
    inputs = []
    labels = []
    text_length = []
    for label, text in batch:
        inputs.append(torch.tensor(text))
        labels.append(int(label))
        text_length.append(len(text))

    inputs = pad_sequence(inputs, batch_first=True, padding_value=1)
    return inputs, torch.tensor(labels), torch.tensor(text_length)


def train(data):
    model.train()
    correct = 0
    total = 0
    avg_loss = 0
    for inputs, labels, input_length in data:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        output = model(inputs, input_length)

        output_prob = output.softmax(dim=2)
        best_class = torch.argmax(output_prob, dim=2)
        predicted = best_class.squeeze().tolist()
        for i in range(len(predicted)):
            if predicted[i] == labels[i]:
                correct += 1
            total += 1

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(1), labels)
        avg_loss += loss
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return correct / total, avg_loss


def evaluate(data):
    model.eval()
    correct = 0
    total = 0
    wrong_predictions = [("label", "prediction", "input text")]
    with torch.no_grad():
        for inputs, labels, input_length in data:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            output = model(inputs, input_length)

            output_prob = output.softmax(dim=2)
            best_class = torch.argmax(output_prob, dim=2)
            predicted = best_class.squeeze().tolist()

            for i in range(len(predicted)):
                input_text_ids = inputs[i]
                input_len = input_length[i]
                input_text = vocab.lookup_tokens(input_text_ids[:input_len].tolist())

                if predicted[i] == labels[i]:
                    correct += 1
                else:
                    wrong_predictions.append((labels[i].item(), predicted[i], " ".join(input_text)))
                total += 1

    return correct / total, wrong_predictions


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, hidden_dim):
        super(Net, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, dropout=0.1, num_layers=2)
        self.linear = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x, input_length):
        embeds = self.embedding_layer(x)  # batch, input, embed
        embeds = self.drop(embeds)
        embeds = pack_padded_sequence(embeds, input_length, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(embeds)  # batch, input length, hidden*2

        lstm_out, out_len = pad_packed_sequence(lstm_out, batch_first=True)

        max_pool = lstm_out.max(dim=1)[0].unsqueeze(1)  # batch, 1, hidden*2

        out = self.linear(max_pool)
        return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--embed-dim', type=int, default=128, metavar='N',
                        help='embedding dimension (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='hidden dimension (default: 128)')
    parser.add_argument('--gd_clip', type=float, default=5,
                        help='gradient clipping (default: 5)')

    args = parser.parse_args()

    batch_size = args.batch_size
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim

    # Training parameters
    epochs = args.epochs
    lr = args.lr
    clip = args.gd_clip  # gradient clipping

    # Read train set and compute vocab
    train_data = read_data("data/sentiment.train.tsv")
    train_texts = []
    label_set = set()
    for label, tokens in train_data:
        tokens = [token.lower() for token in tokens]
        train_texts.append(tokens)
        label_set.add(label)

    vocab = build_vocab_from_iterator(train_texts, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])  # make default index same as index of unk_token
    vocab_size = len(vocab)

    # create a data loader for train
    label_and_token_ids_train = [(label, vocab.lookup_indices(text)) for label, text in train_data]
    train_loader = DataLoader(label_and_token_ids_train, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # create a data loader for dev
    dev_data = read_data("data/sentiment.dev.tsv")
    label_and_token_ids_dev = [(label, vocab.lookup_indices(text)) for label, text in dev_data]
    dev_loader = DataLoader(label_and_token_ids_dev, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # create a data loader for test
    test_data = read_data("data/sentiment.test.tsv")
    label_and_token_ids_test = [(label, vocab.lookup_indices(text)) for label, text in test_data]
    test_loader = DataLoader(label_and_token_ids_test, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print("classes = ", label_set)
    model = Net(vocab_size=vocab_size, embedding_dim=embed_dim, num_classes=len(list(label_set)),
                hidden_dim=hidden_dim).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for i in tqdm(range(1, epochs + 1)):
        print("\nEpoch = ", i)
        train_acc, train_loss = train(train_loader)
        print("Train accuracy = ", train_acc)

        dev_acc, _ = evaluate(dev_loader)
        print("Dev accuracy   = ", dev_acc)
        logger.report_scalar(title='train loss', series='Loss', value=train_loss, iteration=i)
        logger.report_scalar(title='train acc', series='Accuracy', value=train_acc, iteration=i)

        logger.report_scalar(title='dev acc', series='Accuracy', value=dev_acc, iteration=i)

        torch.save(model.state_dict(), "model.pt")

    test_acc, wrong_preds = evaluate(test_loader)
    logger.report_table(title='Wrong Predictions', series='pandas DataFrame', table_plot=wrong_preds)

    logger.report_single_value('test acc', test_acc)
    print("Test accuracy = ", test_acc)

    logger.report_single_value('num train samples', len(train_data))
    logger.report_single_value('num dev samples', len(dev_data))
    logger.report_single_value('num test samples', len(test_data))