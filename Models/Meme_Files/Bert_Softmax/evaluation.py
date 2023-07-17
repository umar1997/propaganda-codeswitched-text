from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer


def get_accuracy(true_tags, pred_tags):
    return accuracy_score(true_tags, pred_tags)*100

def get_f1_score(true_tags, pred_tags):
    #average='micro or macro'
    return f1_score(true_tags, pred_tags, average='weighted', zero_division=1)
    
def get_classification_report(true_tags, pred_tags):
    return classification_report(true_tags, pred_tags)

def get_confusion_matrix(true_tags, pred_tags, labels_):
    print(labels_)
    return confusion_matrix(true_tags, pred_tags, labels=labels_)

def get_MultilabelBinarizer(gold_labels, pred_labels, classes):

    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    gold = mlb.transform(gold_labels)
    pred = mlb.transform(pred_labels)

    macro_f1 = f1_score(gold, pred, average="macro", zero_division=1)
    micro_f1 = f1_score(gold, pred, average="micro", zero_division=1)
    return macro_f1, micro_f1