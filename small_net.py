from clearml import Task
import argparse
import random
import torch
import torch.nn as nn

########## ClearML ##################
Task.add_requirements('requirements.txt')

task = Task.init(
    project_name='ClearML Tutorial',
    task_name='logging examples',
    tags="temp",
    auto_connect_arg_parser=True,
)

task.execute_remotely(queue_name="<=12GB", clone=False, exit_process=True)
logger = task.get_logger()


########## Model #####################

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc2 = nn.Linear(12, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(samples):
    for sample in samples:
        out = model(sample)
    loss = random.random()
    acc = random.random()
    return loss, acc


def dev():
    loss = random.random()
    acc = random.random()
    return loss, acc


def test():
    acc = random.random()
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()

    # main training loop
    model = Net()
    model = model.to(DEVICE)
    train_batches = [torch.randn(5, 12, 32).to(DEVICE) for _ in range(5)]

    for i in range(1, args.epochs + 1):
        train_loss, train_acc = train(train_batches)
        dev_loss, dev_acc = dev()

        # log loss and accuracy
        logger.report_scalar(title='train loss', series='Loss', value=train_loss, iteration=i)
        logger.report_scalar(title='train acc', series='Accuracy', value=train_acc, iteration=i)

        logger.report_scalar(title='dev loss', series='Loss', value=dev_loss, iteration=i)
        logger.report_scalar(title='dev acc', series='Accuracy', value=dev_acc, iteration=i)

    # save model
    torch.save(model.state_dict(), 'model.pt')
    # test
    test_acc = test()
    logger.report_single_value('test acc', test_acc)

    # log number of samples used in the experiment
    logger.report_single_value('num test samples', 1000)
    logger.report_single_value('num train samples', 8000)
    logger.report_single_value('num dev samples', 1000)
