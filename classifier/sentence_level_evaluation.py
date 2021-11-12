import jsonlines
import numpy as np
from classifier.utils import MULTI_LABEL_TO_INDEX, GENERAL_LABEL_TO_INDEX

general_level_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}


def get_eval_metrics(result_fpath, labels_fpath):
    """
    Get some evaluation metrics
    - categorical accuracy
    - accuracy for just three levels A, B, C
    - accuracy per level
    - percentage of samples 1 or less deviation away from true level
    :param result_fpath:
    :param labels_fpath:
    :return:
    """
    labels = []
    texts = []
    num_acc = 0
    num_acc_general_level = 0
    num_less_than_2_deviations_from_true_level = 0
    acc_distribution = {i:{'acc':0, 'inacc':0} for i in range(6)}
    acc_distribution_general = {i: {'acc':0, 'inacc':0} for i in range(3)}
    mean_absolute_error = 0
    with jsonlines.open(labels_fpath) as labels_f:
        for obj in labels_f:
            labels.append(obj['label'])
            #texts.append(obj['text'])
    result_probs = []
    with jsonlines.open(result_fpath) as result:
        for obj in result:
            probs = obj['probs']
            result_probs.append(probs)

    for i in range(len(result_probs)):
        pred_class = np.argmax(result_probs[i])
        actual_class = MULTI_LABEL_TO_INDEX[labels[i]]
        mean_absolute_error += abs(pred_class - actual_class)
        #actual_class = GENERAL_LABEL_TO_INDEX[labels[i]]
        if pred_class == actual_class:
            num_acc += 1
            acc_distribution[actual_class]['acc'] += 1
            # if actual_class == 4:
            #     print(texts[i])
        else:
            acc_distribution[actual_class]['inacc'] += 1
        if abs(pred_class - actual_class) <= 1:
            num_less_than_2_deviations_from_true_level += 1

        general_level_actual = general_level_mapping[actual_class]
        general_level_pred = general_level_mapping[pred_class]
        if general_level_pred == general_level_actual:
            num_acc_general_level += 1
            acc_distribution_general[general_level_actual]['acc'] += 1
        else:
            acc_distribution_general[general_level_actual]['inacc'] += 1

    total_count = len(labels)
    mean_absolute_error /= total_count
    class_acc = num_acc/total_count
    general_level_acc = num_acc_general_level/total_count
    less_than_one_deviation_percent = num_less_than_2_deviations_from_true_level/total_count
    print(f"Class acc: {round(class_acc, 3)}")
    print(f"General Level Acc: {round(general_level_acc, 3)}")
    print(f"Percentage of predictions 1 or less deviation from actual level: {round(less_than_one_deviation_percent, 3)}")
    print(f"Class accuracy distribution: {acc_distribution}")
    print(f"General accuracy distribution: {acc_distribution_general}")
    print(f"Mean absolute error: {round(mean_absolute_error, 3)}")

if __name__ == '__main__':
    test_labels_path = ''
    res_path = ''
    get_eval_metrics(result_fpath=res_path, labels_fpath=test_labels_path)