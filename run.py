import model
import config
import dataset
import active_learning
import query_strategies
import tensorflow as tf
import os
import numpy as np
import sklearn.metrics as metrics
from scipy.spatial.distance import dice

def log(arguments, message):
    """
    Function to handle printing and logging od messages.
    :param arguments: An ArgumentParser object.
    :param message: String with the message to be printed or logged.
    """

    if arguments.verbose:
        print(message)
    if arguments.log_file != '':
        if not os.path.isfile(arguments.log_file):
            if not os.path.isdir(os.path.dirname(arguments.log_file)):
                os.makedirs(os.path.dirname(arguments.log_file))
        print(message, file=open(arguments.log_file, 'a'))


if __name__ == '__main__':
        arguments = config.load_config(description="Active Learning Experiment Framework")

        log(arguments, "Arguments Loaded")

        if arguments.seed != 0:
            np.random.seed(arguments.seed)

        x_train, y_train, x_test, y_test = dataset.get_dataset()

        labeled_indices = np.zeros(len(y_train), dtype=bool)
        labeled_temp = np.arange(len(y_train)) 
        np.random.shuffle(labeled_temp)
        labeled_indices[labeled_temp[:arguments.init_labels]] = True

        model = model.Model('')

        query_strategies = query_strategies.KMeansSampling(x_train, y_train, labeled_indices, model, arguments)

        log(arguments, "\n---------- Iteration 0")
        log(arguments, "Number of initial labeled data: {}".format(list(labeled_indices).count(True)))
        log(arguments, "Number of initial unlabeled data: {}".format(len(y_train) - list(labeled_indices).count(True)))
        log(arguments, "Number of testing data: {}".format(len(y_test)))
        
        jaccard = np.zeros(arguments.num_iterations + 1)
        dice_cof = np.zeros(arguments.num_iterations + 1)
        cm = np.zeros(arguments.num_iterations + 1)
        sensitivity = np.zeros(arguments.num_iterations + 1)
        query_strategies.train()
        predictions, prediction_labels = query_strategies.predict(x_test, y_test)

        jaccard[0] = metrics.jaccard_similarity_score(np.reshape(predictions,[-1,(128 * 128)])[0],np.reshape(y_test[3],[-1, (128 * 128)])[0])
        dice_cof[0] = dice(np.reshape(predictions,[-1,(128 * 128)])[0],np.reshape(y_test[3],[-1, (128 * 128)])[0])
        #cm[0] = metrics.confusion_matrix(np.reshape(predictions,[-1,(128 * 128)])[0],np.reshape(y_test[3],[-1, (128 * 128)])[0])
        #sensitivity[0] = cm[0][0,0]/(cm[0][0,0]+cm[0][0,1])
        log(arguments,"\nJaccard Similarity: {}".format(jaccard[0]))  
        log(arguments, "\nDice: {}".format(dice_cof[0]))
        #log(arguments, "Sensitivity: {}".format(sensitivity[0]))
        # log(arguments, "\nTesting Accuracy: {}".format(accuracy[0]))
        # log(arguments, "Testing Mean-Class Accuracy: {}".format(mca[0]))
        # log(arguments, "Testing Loss: {}\n\n\n".format(losses[0]))

        for iteration in range(1, arguments.num_iterations+1):
            # Runs the specified query method and return the selected indices to be annotated.
            query_indices = query_strategies.query(8)

            # Update the selected indices as labeled.
            labeled_indices[query_indices] = True
            query_strategies.update_labeled_data(labeled_indices)

            # Logs information about the current active learning iteration.
            log(arguments, "\n---------- Iteration " + str(iteration))
            log(arguments, "Number of initial labeled data: {}".format(list(labeled_indices).count(True)))
            log(arguments, "Number of initial unlabeled data: {}".format(len(y_train) - list(labeled_indices).count(True)))
            log(arguments, "Number of testing data: {}".format(len(y_test)))

            # Train the model with the new training set.
            query_strategies.train()

            # Get the predictions from the testing set.
            predictions, prediction_labels = query_strategies.predict(x_test, y_test)
            jaccard[iteration] = metrics.jaccard_similarity_score(np.reshape(predictions,[-1,(128 * 128)])[0],np.reshape(y_test[3],[-1, (128 * 128)])[0])
            dice_cof[iteration] = dice(np.reshape(predictions,[-1,(128 * 128)])[0],np.reshape(y_test[3],[-1, (128 * 128)])[0])
            #cm[iteration] = metrics.confusion_matrix(np.reshape(predictions,[-1,(128 * 128)])[0],np.reshape(y_test[3],[-1, (128 * 128)])[0])
            #sensitivity[iteration] = cm[iteration][0,0]/(cm[iteration][0,0]+cm[0][0,1])
            # Calculates the accuracy of the model based on the model's predictions.
            # accuracy[iteration] = 1.0 * (y == prediction_labels).sum().item() / len(y)
            # cmat = metrics.confusion_matrix(y, prediction_labels)
            # mca[iteration] = np.mean(cmat.diagonal() / cmat.sum(axis=1))
            # losses[iteration] = torch.nn.functional.cross_entropy(predictions, y).item()

            # Logs the testing accuracy.
            log(arguments, "\nJaccard Similarity: {}".format(jaccard[iteration]))
            log(arguments, "\nDice: {}".format(dice_cof[iteration]))
            #log(arguments, "Sensitivity: {}".format(sensitivity[iteration]))
            #log(arguments, "\nTesting Accuracy: {}".format(accuracy[iteration]))
            #log(arguments, "Testing Mean-Class Accuracy: {}".format(mca[iteration]))
            #log(arguments, "Testing Loss: {}\n\n\n".format(losses[iteration]))

        # Logs the accuracies from all iterations.
        log(arguments, "Jaccard Similarity: " + str(np.mean(jaccard)))
        log(arguments, "Dice: " + str(np.mean(dice_cof)))
        #log(arguments, "Dice: " + str(np.mean(sensitivity)))
        # log(arguments, "Accuracy: " + str(accuracy))
        # log(arguments, "Mean-Class Accuracy: " + str(mca))
        # log(arguments, "Loss: " + str(losses))
        log(arguments, "\n\n\n\n\n\n\n\n\n\n")
