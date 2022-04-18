import time
import numpy as np
import random

import matplotlib.pyplot as plt


def load_csv(path):
    '''
    Load the CSV form of MNIST data without any external library
    :param path: the path of the csv file
    :return:
        data: A list of list where each sub-list with 28x28 elements
              corresponding to the pixels in each image
        labels: A list containing labels of images
    '''
    data = []
    labels = []
    with open(path, 'r') as fp:
        images = fp.readlines()
        images = [img.rstrip() for img in images]

        for img in images:
            img_as_list = img.split(',')
            y = int(img_as_list[0]) # first entry as label
            x = img_as_list[1:]
            x = [int(px) / 255 for px in x]
            data.append(x)
            labels.append(y)
    return data, labels


def load_mnist_trainval():
    """
    Load MNIST training data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        val_data: A list of list containing the validation data
        val_label: A list containing the labels of validation data
    """
    # Load training data
    print("Loading training data...")
    data, label = load_csv('./data/mnist_train.csv')
    assert len(data) == len(label)
    print("Training data loaded with {count} images".format(count=len(data)))

    # split training/validation data
    train_data = None
    train_label = None
    val_data = None
    val_label = None
    #############################################################################
    # TODO:                                                                     #
    #    1) Split the entire training set to training data and validation       #
    #       data. Use 80% of your data for training and 20% of your data for    #
    #       validation                                                          #
    #############################################################################
    N = len(data)
    p = 0.8
    train_data = data[0:int(N*p)]
    train_label = label[0:int(N*p)]
    val_data = data[int(N*p):]
    val_label = label[int(N*p):]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return train_data, train_label, val_data, val_label

def load_mnist_test():
    """
        Load MNIST testing data with labels
        :return:
            data: A list of list containing the testing data
            label: A list containing the labels of testing data
        """
    # Load training data
    print("Loading testing data...")
    data, label = load_csv('./data/mnist_test.csv')
    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    return data, label



def generate_batched_data(data, label, batch_size=32, shuffle=False, seed=None):
    '''
    Turn raw data into batched forms
    :param data: A list of list containing the data where each inner list contains 28x28
                 elements corresponding to pixel values in images: [[pix1, ..., pix784], ..., [pix1, ..., pix784]]
    :param label: A list containing the labels of data
    :param batch_size: required batch size
    :param shuffle: Whether to shuffle the data: true for training and False for testing
    :return:
        batched_data: A list whose elements are batches of images.
        batched_label: A list whose elements are batches of labels
    '''
    batched_data = np.array([])
    batched_label = np.array([])
    if seed:
        random.seed(seed)
    #############################################################################
    # TODO:
    #    1) Shuffle data using random.shuffle and label if shuffle=True                              #
    #    2) Generate batches of images with the required batch size             #
    #    It's okay if the size of your last batch is smaller than the required  #
    #    batch size                                                             #
    #############################################################################

    data = np.array(data)
    label = np.array(label)
    # N1, N2 = np.array(data).shape
    N1 = len(data)
    # n_batches = int(np.ceil(float(N1)/float(batch_size)))

    if shuffle:
        index = list(range(len(data)))
        random.shuffle(index)
        data = data[index]
        label = label[index]

    # print("\n\n ==========")
    # print("N = ", N1)
    # print("N2 = ", N2)
    # print("batch_size = ", batch_size)
    # print("n_batches = ", n_batches)

    ix = np.arange(batch_size, N1, batch_size)

    batched_data = np.split(data, ix, 0)
    batched_label = np.split(label, ix, 0)

    # print(len(batched_data))
    # print(len(batched_label))

    # print(batched_data[1].shape)
    # print(batched_label[1].shape)
    # print("Second to last row:")
    # print(batched_data[-2].shape)
    # print(batched_label[-2].shape)
    # print("Last row:")
    # print(batched_data[-1].shape)
    # print(batched_label[-1].shape)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return batched_data, batched_label


def train(epoch, batched_train_data, batched_train_label, model, optimizer, debug=True):
    '''
    A training function that trains the model for one epoch
    :param epoch: The index of current epoch
    :param batched_train_data: A list containing batches of images
    :param batched_train_label: A list containing batches of labels
    :param model: The model to be trained
    :param optimizer: The optimizer that updates the network weights
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    '''
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_train_data, batched_train_label)):

        start_time = time.time()
        loss, accuracy = model.forward(input, target)

        optimizer.update(model)
        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]

        forward_time = time.time() - start_time
        if idx % 10 == 0 and debug:
            print(('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f} \t'
                  'Batch Loss {loss:.4f}\t'
                  'Train Accuracy ' + "{accuracy:.4f}" '\t').format(
                epoch, idx, len(batched_train_data), batch_time=forward_time,
                loss=loss, accuracy=accuracy))
    epoch_loss /= len(batched_train_data)
    epoch_acc = hits / count_samples

    if debug:
        print("* Average Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))
    return epoch_loss, epoch_acc



def evaluate(batched_test_data, batched_test_label, model, debug=True):
    '''
    Evaluate the model on test data
    :param batched_test_data: A list containing batches of test images
    :param batched_test_label: A list containing batches of labels
    :param model: A pre-trained model
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    '''
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):

        loss, accuracy = model.forward(input, target, mode='valid')

        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]
        if debug:
            print(('Evaluate: [{0}/{1}]\t'
                  'Batch Accuracy ' + "{accuracy:.4f}" '\t').format(
                idx, len(batched_test_data), accuracy=accuracy))
    epoch_loss /= len(batched_test_data)
    epoch_acc = hits / count_samples

    return epoch_loss, epoch_acc


def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history):
    '''
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    '''
    #############################################################################
    # TODO:                                                                     #
    #    1) Plot learning curves of training and validation loss                #
    #    2) Plot learning curves of training and validation accuracy            #
    #############################################################################
    plt.subplot(1,2,1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.plot(train_loss_history, label = "train")
    plt.plot(valid_loss_history, label = "valid")
    plt.legend()

    plt.subplot(1,2,2)
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Acc Curve')
    plt.plot(train_acc_history, label = "train")
    plt.plot(valid_acc_history, label = "valid")
    plt.legend()
    
    plt.savefig("plots.png")
    # plt.show()
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################