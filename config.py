from pathlib import Path

EXP_NAME = 'MLP'
EPOCH = 20
EMBEDDING_DIM = 64
ENCODER_STACK = 6
ATTENTION_HEAD = 1
DROPOUT = 0.1
LR = 0.0004
BATCH_SIZE = 32
AUGMENTATION = 55

MODEL_DIR = Path.cwd() / "models" / EXP_NAME

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)

FEATURES = ['Baseline Features', 'Intensity Parameters', 'Formant Frequencies', 'Bandwidth Parameters',
                  'Vocal Fold', 'MFCC', 'Wavelet Features', 'TQWT Features']

FEATURE_GROUPS = ['Basic Info', 'Baseline Features', 'Intensity Parameters', 'Formant Frequencies',
                  'Bandwidth Parameters', 'Vocal Fold', 'MFCC', 'Wavelet Features', 'TQWT Features']
