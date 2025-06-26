# Mask pixel values for different bounding boxes
MASK_LABELS = {
    '1': {'mito_label': 5, 'vesicle_label': 6, 'cleft_label': 7, 'cleft_label2': 7},  # default case
    '2': {'mito_label': 1, 'vesicle_label': 3, 'cleft_label2': 4, 'cleft_label': 2},
    '3': {'mito_label': 6, 'vesicle_label': 7, 'cleft_label2': 8, 'cleft_label': 9},
    '4': {'mito_label': 3, 'vesicle_label': 2, 'cleft_label2': 4, 'cleft_label': 1},
    '5': {'mito_label': 1, 'vesicle_label': 3, 'cleft_label2': 4, 'cleft_label': 2},
    '6': {'mito_label': 5, 'vesicle_label': 6, 'cleft_label': 7, 'cleft_label2': 7},  # default case
    '7': {'mito_label': 1, 'vesicle_label': 2, 'cleft_label2': 3, 'cleft_label': 4},
}

def get_mask_labels(bbox_num):
    return MASK_LABELS.get(bbox_num, {'mito_label': 5, 'vesicle_label': 6, 'cleft_label': 7, 'cleft_label2': 7}) 