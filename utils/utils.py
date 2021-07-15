import pickle


def filter_feature(feature_frame, feature_score_file, max_features=32):
    with open(feature_score_file, 'rb') as handle:
        scores = pickle.load(handle)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    to_keep = [col for col, _ in scores[:max_features]]
    to_keep.append('class')
    feature_frame = feature_frame[to_keep]
    return feature_frame