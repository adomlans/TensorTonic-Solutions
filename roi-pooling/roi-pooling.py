import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    # Write code here
    import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    results = []
    for roi in rois:
        x1, y1, x2, y2 = roi
        rh, rw = y2 - y1, x2 - x1
        pooled = []
        for i in range(output_size):
            row = []
            for j in range(output_size):
                hs = y1 + int(math.floor(i * rh / output_size))
                he = y1 + int(math.floor((i + 1) * rh / output_size))
                ws = x1 + int(math.floor(j * rw / output_size))
                we = x1 + int(math.floor((j + 1) * rw / output_size))
                he, we = max(he, hs + 1), max(we, ws + 1)
                mx = float("-inf")
                for r in range(hs, he):
                    for c in range(ws, we):
                        if 0 <= r < len(feature_map) and 0 <= c < len(feature_map[0]):
                            mx = max(mx, feature_map[r][c])
                row.append(mx)
            pooled.append(row)
        results.append(pooled)
    return results

    pass