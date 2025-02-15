import argparse
import time
import math
import os

from tqdm import tqdm
import torch

from data_utils import get_lm_corpus
from utils.exp_utils import get_logger
from mem_transformer import MemTransformerLM

parser = argparse.ArgumentParser(
    description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--kernel_size', type=int, nargs='+', default=[1, 1],
                        help='kernel size of block matrices in PatchAttn.')
parser.add_argument('--stride', type=int, nargs='+', default=[1, 1],
                        help='stride of block matrices in PatchAttn.')
parser.add_argument('--skip_first', action='store_true',
                    help='old behavior, skip tgt_len first tokens')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
parser.add_argument('--M-positions', nargs = '+', type = int, default = [],
                    help='List of positions for M-attention')
parser.add_argument('--show-M', action = "store_true",
                    help='Show Mahalanobis transformation matrix.')
parser.add_argument('--downsample-size', type = int, default = 0,
                    help = 'Amount of downsampling to do in M computation, recommended to choose multiples \
                    of d_head. Must be less than sequence length. 0 for no downsampling i.e using all queries/keys')
parser.add_argument('--compare-downsample-grads', action = "store_true", help = 'compare downsampled average gradients to fully estimated gradients')
parser.add_argument('--smoothing-correction', type = float, default = 1.0, help = 'amplify the bandwidth/softmax temp in the compute H_hat function')
parser.add_argument('--over-layers', action = 'store_true', help = 'estimate grads using v and q over layers')
parser.add_argument('--attenuation', type = float, default = 5e-2, help = 'attenuation factor in W_over_layers computation')
parser.add_argument('--ablation', action = 'store_true')
parser.add_argument('--grad-median', action = 'store_true', help = 'use median when average L1 norm of gradients for robustness')
parser.add_argument('--output-dir', type = str, default = '')
parser.add_argument('--model-size', type = str)

args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.output_dir, 'log_eval_attack.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

# Load the best saved model.
# model params
n_layer = 16
n_head = 8 
d_model = 128 if args.model_size == 'small' else 256
d_head = 16 if args.model_size == 'small' else 32
d_inner = 2048
tgt_len = 256 if args.model_size == 'small' else 384
eval_tgt_len = 256 if args.model_size == 'small' else 384
dropout = 0.1
dropatt = 0.0
tie_weight = True
d_embed = d_model
div_val = 1 
pre_lnorm= False
same_length= False
attn_type= 2
sample_softmax= -1
proj_dim= 16
n_roll=2
skip_attn_normalization=False
no_pos=False
device=device
update_mode= 'hard'
n_global_head= 2

# in rachel code clamp len set to 256, double check this isn't different to -1 setting.


cutoffs, tie_projs = [], [False]
cutoffs = [20000, 40000, 200000]
tie_projs += [True] * len(cutoffs)

model = MemTransformerLM(
        ntokens,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        M_positions = args.M_positions,
        show_M = args.show_M,
        over_layers= args.over_layers,
        attenuation=args.attenuation,
        median = args.grad_median,
        ablation = args.ablation,
        tie_weight=tie_weight,
        d_embed=d_embed,
        div_val=div_val,
        tie_projs=tie_projs,
        pre_lnorm=pre_lnorm,
        tgt_len=tgt_len,
        ext_len=args.ext_len,
        mem_len=args.mem_len,
        cutoffs=cutoffs,
        same_length=same_length,
        attn_type=attn_type,
        clamp_len=args.clamp_len,
        sample_softmax=sample_softmax,
        proj_dim=proj_dim,
        n_roll= n_roll,
        skip_attn_normalization=skip_attn_normalization,
        no_pos=no_pos,
        device=device,
        update_mode=update_mode,
        kernel_size=args.kernel_size, 
        stride=args.stride,
        n_global_head=n_global_head
    )

with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    print(f'Loading checkpoint from: {f.name}')
    model = torch.load(f)
model.backward_compatible()
model = model.to(device)

if not args.over_layers: # when over-layers is false, we're using the older method1 models. Need to assign in the new attributes
    for i, layer in enumerate(model.layers):
        layer.over_layers = False
        layer.smoothing_correction = 1.
        layer.dec_attn.over_layers = False
        layer.dec_attn.smoothing_correction = 1.


logging(f'Evaluating with bsz {args.batch_size} tgt_len {tgt_len} '
        f'ext_len {args.ext_len} mem_len {args.mem_len} '
        f'clamp_len {args.clamp_len} using a sliding window.')


model.reset_length(tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True
assert model.same_length is False

skip_first = args.skip_first


###############################################################################
# Evaluation code
###############################################################################
def evaluate(eval_iter):
    global skip_first
    print(f"old behavior: {skip_first}")
    # Turn on evaluation mode which disables dropout.
    model.eval()
    
    total_len, total_loss = 0, 0.
    start_time = time.time()
    with torch.no_grad():
        mems = tuple()
        with tqdm(total=len(eval_iter)) as progress_bar:
            for idx, (data, target, _) in enumerate(eval_iter):
                ret = model(data, target, *mems, softmax_keep_order=True)
                bsz = target.size(1)
                loss, mems = ret[0], ret[1:]
                if skip_first:
                    # TODO add option in the model to skip computing softmax
                    # for all but the last position.
                    # shape of loss: (len,B)
                    # take only the last position, be careful with
                    # proj_adaptive_softmax which can change the indices
                    # if softmax_keep_order is False.
                    loss = loss[-1].sum(dim=-1)  # mean across batch dim
                    total_loss += loss.item()
                    total_len += bsz
                else:
                    if idx == 0:
                        total_len += loss.shape[0] * loss.shape[1]
                        loss = loss.sum()
                        total_loss += loss.item()
                    else:
                        # TODO add option in the model to skip computing
                        # softmax for all but the last position.
                        # shape of loss: (len,B)
                        # take only the last position, be careful with
                        # proj_adaptive_softmax which can change the indices
                        # if softmax_keep_order is False.
                        loss = loss[-1].sum(dim=-1)  # mean across batch dim
                        total_loss += loss.item()
                        total_len += bsz
                progress_bar.update(1)
        total_time = time.time() - start_time
    logging(f'{total_len} positions evaluated.')
    logging(f'Time : {total_time :.2f}s, '
            f'{ 1000 * total_time / (idx+1):.2f}ms/segment')
    return total_loss / total_len


# Run on test data.
if args.split == 'all':
    va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                                  device=device,
                                  ext_len=args.ext_len, sliding_window=True)
    te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                                  device=device,
                                  ext_len=args.ext_len, sliding_window=True)
    test_loss = evaluate(te_iter)
    valid_loss = evaluate(va_iter)
elif args.split == 'valid':
    va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len,
                                  sliding_window=True)
    valid_loss = evaluate(va_iter)
    test_loss = None
elif args.split == 'test':
    te_iter = corpus.get_iterator('test', args.batch_size, tgt_len,
                                  device=device, ext_len=args.ext_len,
                                  sliding_window=True)
    test_loss = evaluate(te_iter)
    valid_loss = None


def format_log(loss, split):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str


log_str = ''
if valid_loss is not None:
    log_str += format_log(valid_loss, 'valid')
if test_loss is not None:
    log_str += format_log(test_loss, 'test')

logging('=' * 100)
logging(log_str)
logging('=' * 100)
