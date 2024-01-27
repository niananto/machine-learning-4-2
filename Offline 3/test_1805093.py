import seaborn as sns
import os
import re
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from util_1805093 import *

# read the test dataset
independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

# convert the test dataset to numpy arrays
X_test, y_test = to_ndarray(independent_test_dataset)

def test(filename, is_best=False):
    # read the pickle file
    with open(filename, "rb") as f:
        model = pickle.load(f)
        
    # print the test accuracy
    print("Test accuracy:", np.mean(np.argmax(predict(model, X_test), axis=1) == np.argmax(y_test, axis=1)))

    conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predict(model, X_test), axis=1))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues', cbar=False)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')
    if not is_best:
        id = filename.split('_')[1]
        lr = str.join('', (filename.split('_')[2].split('.')[:1]))
        print(f'model {id} with learning rate {lr}')
        plt.savefig(f'conf_mats/conf_{id}_{lr}.png')
    else:
        plt.savefig('confusion_matrix.png')
    plt.clf()

    print(classification_report(np.argmax(y_test, axis=1), np.argmax(predict(model, X_test), axis=1)))
    print("F1 score:", f1_score(np.argmax(y_test, axis=1), np.argmax(predict(model, X_test), axis=1), average='macro'))
    
# run test method on all pickle files in the pickles folder
for filename in os.listdir('pickles'):
    test('pickles/' + filename)
    
# test the best model
test('model_1805093.pickle', is_best=True)