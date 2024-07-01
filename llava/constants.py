CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
VISUAL_LENGTH = 576

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

import itertools
import torch
# create a look up mapping table for the image tokens 576 -> 24 x 24
MAPPING = list(itertools.product(range(24), range(24)))
MAPPINGX = torch.tensor([x for _, x in MAPPING])
MAPPINGY = torch.tensor([y for y, _ in MAPPING])

COLOR_CHOICES = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'white', 'black', 'gray']
YES_NO_CHOICES = ['yes', 'no']
NUMBER_CHOICES_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SIZE_CHOICES = ['small', 'large']
NUMBER_CHOICES = [str(x) for x in NUMBER_CHOICES_]

if __name__ == "__main__":
    print(",".join(NUMBER_CHOICES)) 