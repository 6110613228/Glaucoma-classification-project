import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model('VGG16_epochs_35_lr_e-4.h5')

# Load test_dataset with shuffle=False
test_path = 'test_dataset'
test_batches = ImageDataGenerator().flow_from_directory(
    test_path, target_size=(224, 224), shuffle=False
)

# Load label of each data
y_test = test_batches.classes

# Let model predict our dataset
y_pred = model.predict(test_batches)

# Calculate ROC_Curve for each class against non-class
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:, 0], pos_label=0)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test, y_pred[:, 1], pos_label=1)
fpr3, tpr3, thresholds3 = metrics.roc_curve(y_test, y_pred[:, 2], pos_label=2)

# Plot ROC_Curve for each classes
roc_auc = auc(fpr, tpr)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)

plt.figure(1)
lw = 2
plt.plot(fpr, tpr, lw=lw, label='ROC curve Glaucoma (area = %.4f)' % roc_auc)
plt.plot(fpr2, tpr2, lw=lw, label='ROC curve Normal (area = %.4f)' % roc_auc2)
plt.plot(fpr3, tpr3, lw=lw, label='ROC curve Other (area = %.4f)' % roc_auc3)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

y_predict = y_pred.argmax(axis=1)

# Show classification report
print(metrics.classification_report(y_test, y_predict, digits=5))

# Calculate confusion_matrix
cf_matrix_3x3 = metrics.confusion_matrix(y_test, y_predict)

# Plot confusion_matrix
make_confusion_matrix(cf_matrix_3x3, figsize=(8,6), cbar=False)

# Function for plot confusion_matrix
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=False,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(
            value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(
        group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
