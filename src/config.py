from datetime import datetime
from pytz import timezone
import torch

class config:
    n_fold = 1
    epochs = 15
    image_size = (512, 512)
    hop_length = 24
    use_clip = True
    clip_rate = 3.5
    use_amp = True
    freeze = False
    early_stop = True
    model_name = "tf_efficientnetv2_b0"
    training_date = str(datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d'))
    weight_path = "weights/" + training_date
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    gradient_accumulation_steps = 1
    num_classes = 1
    lr = 1e-4
    num_workers = 4
    use_roc_star = False
    do_mixup = False
    alpha = 0.4 # mixup
    log_file_name = "logs/" + str(datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d')) + ".log"