# DATA
dataset='CULane'
data_root = '/mnt/HD/dataset/CULane/'

# TRAIN
epoch = 80
batch_size = 12
optimizer = 'AdamW'  #['SGD','Adam']
learning_rate = 8e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

lambda_1 = 10
lambda_2 = 1

# NETWORK
use_aux = False
row_num = 128
griding_num = 256

# EXP
note = ''

log_path = './experiment'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = "/mnt/SDC/madongliang/test/ERF-E2E/experiment/20200915_113049_lr_8e-04_b_12/ep054.pth"
test_work_dir = "./result"

thresh_vc = 0.6
thresh_lc = 0.5













