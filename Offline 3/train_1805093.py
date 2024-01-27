import shutil
import gc
from util_1805093 import *

######################################################################
# DATA LOADING

train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# plt.imshow(train_validation_dataset[1][0][0], cmap='gray')
# plt.show()

train_val_images, train_val_labels = to_ndarray(train_validation_dataset)
X_train, X_val, y_train, y_val = train_test_split(train_val_images, train_val_labels, test_size=0.15, random_state=93)

# print(X_train.shape, y_train.shape)
# print(X_val.shape, y_val.shape)

######################################################################
# PARAMS

np.random.seed(112)

learning_rates = [0.005, 0.001, 0.0005]
iterations = [
    {
        'id' : 1,
        'epochs' : 20,
        'batch_size' : 1024,
        'model' : [
            Dense(784, 128),
            Dropout(0.5),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, 26),
            Softmax()
        ],
    },
    {
        'id' : 2,
        'epochs' : 30,
        'batch_size' : 1024,
        'model' : [
            Dense(784, 1024),
            Dropout(0.5),
            ReLU(),
            Dense(1024, 26),
            ReLU(),
            Softmax()
        ],
    },
    {
        'id' : 3,
        'epochs' : 50,
        'batch_size' : 1024,
        'model' : [
            Dense(784, 1024),
            Dropout(0.5),
            ReLU(),
            Dense(1024, 512),
            ReLU(),
            Dense(512, 26),
            Softmax()
        ],
    },
]      
        
max_accuracy = 0
for iter in iterations:
    id = iter['id']
    epochs = iter['epochs']
    batch_size = iter['batch_size']
    model = iter['model']
    for learning_rate in learning_rates:
        print(f'ID: {id}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}')
        training_loss, validation_loss, training_accuracy, validation_accuracy, training_f1, validation_f1 = train(model, 
                                                                                    cross_entropy, cross_entropy_prime, 
                                                                                    X_train, y_train, X_val, y_val,
                                                                                    epochs, learning_rate, batch_size, verbose=True)
        
        val_acc = np.mean(np.argmax(predict(model, X_val), axis=1) == np.argmax(y_val, axis=1))
        print("Validation accuracy:", val_acc)

        # save important variables in a pickle file
        file = open(f'pickles/model_{id}_{learning_rate}.pickle', 'wb')
        pickle.dump(model, file)
        file.close()
        
        # save the most accurate model separately
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            shutil.copy(f'pickles/model_{id}_{learning_rate}.pickle', 'model_1805093.pickle')
            
        # plot the losses, accuracies and f1 scores for training and validation sets
        # let's show these plots side by side
        f, axarr = plt.subplots(1, 3, figsize=(15,5))
        axarr[0].set_ylim([0, 1])
        axarr[1].set_ylim([0, 1])
        axarr[2].set_ylim([0, 1])
        
        axarr[0].plot(training_loss, label='Training')
        axarr[0].plot(validation_loss, label='Validation')
        axarr[0].set_title('Loss')
        axarr[0].legend()
        axarr[1].plot(training_accuracy, label='Training')
        axarr[1].plot(validation_accuracy, label='Validation')
        axarr[1].set_title('Accuracy')
        axarr[1].legend()
        axarr[2].plot(training_f1, label='Training')
        axarr[2].plot(validation_f1, label='Validation')
        axarr[2].set_title('F1 score')
        axarr[2].legend()
        plt.savefig(f'plots/plot_{id}_{learning_rate}.png')
        plt.clf()
        
        gc.collect()