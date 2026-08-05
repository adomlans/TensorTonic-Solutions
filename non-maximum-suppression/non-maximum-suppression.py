def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    if not boxes:
        return []
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = []
    while indices:
        c = indices.pop(0)
        kept.append(c)
        remaining = []
        for i in indices:
            x1 = max(boxes[c][0], boxes[i][0])
            y1 = max(boxes[c][1], boxes[i][1])
            x2 = min(boxes[c][2], boxes[i][2])
            y2 = min(boxes[c][3], boxes[i][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (boxes[c][2] - boxes[c][0]) * (boxes[c][3] - boxes[c][1])
            a2 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) else 0.0
            if iou < iou_threshold:
                remaining.append(i)
        indices = remaining
    return kept
