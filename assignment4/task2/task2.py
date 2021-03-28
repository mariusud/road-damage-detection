import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    xA = max(prediction_box[0], gt_box[0])
    yA = max(prediction_box[1], gt_box[1])
    
    xB = min(prediction_box[2], gt_box[2])
    yB = min(prediction_box[3], gt_box[3])

    predicted_area = (prediction_box[2] - prediction_box[0] ) * (prediction_box[3] - prediction_box[1])
    
    gt_area = ((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))

    # Compute intersection

    intersection = max(0,xB - xA) * max(0,yB - yA)

    # Compute union
    iou = intersection / float(predicted_area+gt_area-intersection)
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp):
        precision = num_tp / (num_tp + num_fp)
        return precision
    else:
        return 1


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn):
        recall = num_tp / (num_tp + num_fn)
        return recall
    else:
        return 0



def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    preds = []
    gt = []

    for gt_box in gt_boxes:
        gt_box_match = None
        gt_box_best_iou = 0
        for prediction_box in prediction_boxes:
            iou = calculate_iou(prediction_box,gt_box)
            if iou >= iou_threshold and iou > gt_box_best_iou:
                gt_box_match = prediction_box
                gt_box_best_iou = iou
            else:
                #no match
                pass
        if gt_box_match is not None:
            preds.append(gt_box_match)
            gt.append(gt_box)

    # Sort all matches on IoU in descending order
    # Find all matches with the highest IoU threshold
    return np.array(preds), np.array(gt)


    if  len(prediction_boxes)==0 or len(gt_boxes)==0 :
        return np.array([]), np.array([])

    m=gt_boxes.shape[0]
    n=gt_boxes.shape[1]
    count=0
    dict_IoU ={}
    matched_box =np.zeros((m,n))
    predicted_box=np.zeros((m,n))
    for i in range(gt_boxes.shape[0]):
        for k in range(prediction_boxes.shape[0]) :
            iou_real = calculate_iou(prediction_boxes[k,:],gt_boxes[i,:])
            if iou_real>= iou_threshold :
                dict_IoU[(i,k)]= iou_real
    while dict_IoU:
        best_match = max(dict_IoU.keys(), key=lambda key: dict_IoU[key])
        temp = dict_IoU.copy()
        for key in temp.keys():
            if key[0]==best_match[0] or key[1]==best_match[1]:
                dict_IoU.pop(key)
        matched_box[count,:] = gt_boxes[best_match[0],:]
        predicted_box[count,:] = prediction_boxes[best_match[1],:]
        count+=1
    if m!=count :
        index = m-count
        return predicted_box[:-index, :],matched_box[:-index, :]
    else:
        return predicted_box,matched_box


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    preds, gt = get_all_box_matches(prediction_boxes,gt_boxes,iou_threshold)
    return {
        "true_pos": len(preds),
        "false_pos": len(prediction_boxes) - len(preds),
        "false_neg": len(gt_boxes) - len(preds)
    }    


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    TP, FP, FN = 0, 0, 0
    for preds, gt in zip(all_prediction_boxes, all_gt_boxes):
        individual_dict = calculate_individual_image_result(preds,gt,iou_threshold)
        TP +=individual_dict['true_pos']
        FP +=individual_dict['false_pos']
        FN +=individual_dict['false_neg']
    return (calculate_precision(TP,FP,FN), calculate_recall(TP,FP,FN))


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions, recalls = [], []

    for threshold in confidence_thresholds:
        predictions = []
        for i, prediction_box in enumerate(all_prediction_boxes):
            confidence_predictions = []
            for j, box in enumerate(prediction_box):
                if confidence_scores[i][j] >= threshold:
                    confidence_predictions.append(box)
            predictions.append(np.array(confidence_predictions))

        precision, recall = calculate_precision_recall_all_images(predictions,all_gt_boxes,iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    precision, interpolations  = [], []
    for recall_level in recall_levels:
        for inter_precision, recall in zip(precisions, recalls):
            if recall >= recall_level:
                interpolations.append(inter_precision)
        if interpolations:
            precision.append(max(interpolations))
        else:
            precision.append(0)
        interpolations = []
    return np.average(np.array(precision))


    

def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
