import json


def filter_feature(feature_frame, feature_score_file, max_features=32):
    with open(feature_score_file, 'r', encoding='utf8') as handle:
        scores = list(json.load(handle).items())
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    to_keep = [col for col, _ in scores[:max_features]]
    to_keep.append('class')
    to_keep = ['id'] + to_keep
    feature_frame = feature_frame[to_keep]
    return feature_frame

def id_to_index(feature_frame, ids):
    feature_frame['valid'] = feature_frame.apply(lambda x: x['id'] in ids, axis=1)
    feature_frame = feature_frame[feature_frame['valid']]
    return list(feature_frame.index)