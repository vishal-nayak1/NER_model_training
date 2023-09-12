# import logging
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import random
import warnings
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES
from datasets import load_metric
from transformers import get_linear_schedule_with_warmup
from transformers import LayoutLMForTokenClassification

from tqdm import tqdm
import numpy as np
from transformers import AdamW
from utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def set_seed(seed): ## for reproductibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def do_eval(model, dataloader_eval, device):
    eval_loss = 0.0
    tmp_eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    # put model in evaluation mode
    model.eval()
    for batch in tqdm(dataloader_eval, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels)
            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id: #escludi pad e other
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    strict = classification_report(out_label_list, preds_list,  output_dict=True, mode='strict', scheme=IOBES)

    strict['eval_loss'] = eval_loss / nb_eval_steps
    return strict


def train_layoutLM(model, epochs, dataloader_train, dataloader_eval, optimizer, scheduler, early_stop_arg, run, test_mode, seed, saved_checkpoint_dir, model_outputs, best_model_dir, device):
    #args for early stop
    last_loss = 1000
    last_f1 = 0
    patience = early_stop_arg['patience']
    trigger_times = 0

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.zero_grad()

    final_results = []
    # steps = -1
    global_step = 0
    num_epochs = epochs
    set_seed(seed)
    # model.train()
    for epoch in range(1, num_epochs):  # loop over the dataset multiple times
        nb_train_steps = 0
        tr_loss = 0.0
        for steps, batch in enumerate(tqdm(dataloader_train, desc=f'training {epoch} / {num_epochs}')):
            model.train()
            # get the inputs;
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            # optimizer.zero_grad()
            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels)

            loss = outputs.loss
            tr_loss += loss.item()
            loss.backward()

            # if (steps+1) % 100 == 0:
            #     print(f"Train Epoch : {steps+1}/{len(train_dataloader)}")

            if (steps + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
            nb_train_steps += 1

        # print(f"Loss after {epoch} epochs: {loss.item()}")
        total_trn_loss = tr_loss / nb_train_steps
        logger.info("Total Average Train Loss after %s epochs: %s" %(epoch, total_trn_loss))
        eval_results = do_eval(model, dataloader_eval, device)
        #print(f'Validation results: {eval_results}')
        current_loss = eval_results['eval_loss']
        current_f1 = eval_results['micro avg']['f1-score']
        logger.info('Validaiton loss: %s' %(current_loss))
        logger.info('F1 score: %s' %(current_f1))
        #implementing early stopping
        if test_mode == 'val_loss':
            if current_loss > last_loss:
                trigger_times += 1
                logger.info('Validation loss did not decrease from %s' %(last_loss))
                logger.info('Trigger Times: %s' %(trigger_times))

                if trigger_times >= patience or epoch == num_epochs - 1:
                    logger.info('Early stopping because validation loss did not decrease after %s epochs.' %(trigger_times))
                    logger.info('Returning best model named: %s' %(best_model))
                    best_model = torch.load(best_model)
                    df = pd.DataFrame(final_results)
                    if not os.path.exists(f'{model_outputs}'):
                        os.mkdir(f'{model_outputs}')
                    df.to_csv(f'{model_outputs}/log_v1_FUNSD_{test_mode}_run{run}.csv', index = False)
                    return best_model

            else:
                logger.info('Validation loss decresed from %s. Saving checkpoint...' %(last_loss))
                best_model = f'{best_model_dir}/checkpointLM1_epoch{epoch}.pt'
                for ckpt in os.listdir(f'{best_model_dir}'):
                    if 'checkpointLM1_epoch' in ckpt:
                        os.remove(f'{best_model_dir}/{ckpt}') #avoid too many checkpoints
                logger.info('Saving best model : %s' % (best_model))
                torch.save(model, best_model)
                logger.info('Saving model checkpoint at : %s' % (saved_checkpoint_dir))
                model.save_pretrained(saved_checkpoint_dir)
                trigger_times = 0
                last_loss = current_loss
        elif test_mode == 'f1_score':
            if current_f1 < last_f1:
                trigger_times += 1
                logger.info('f1 score did not increase from %s.' %(last_f1))
                logger.info('Trigger Times: %s' %(trigger_times))

                if trigger_times >= patience or epoch == num_epochs - 1:
                    logger.info('Early stopping because f1_score did not increase after %s epochs.' %(trigger_times))
                    logger.info('Returning best model named: %s' %(best_model))
                    best_model = torch.load(best_model)
                    df = pd.DataFrame(final_results)
                    if not os.path.exists(f'{model_outputs}'):
                        os.mkdir(f'{model_outputs}')
                    df.to_csv(f'{model_outputs}/log_v1_FUNSD_{test_mode}_run{run}.csv', index = False)
                    return best_model

            else:
                logger.info('F1 score incresead from %s. Saving checkpoint...' %(last_f1))
                best_model = f'{best_model_dir}/checkpointLM1_epoch{epoch}.pt'
                for ckpt in os.listdir(f'{best_model_dir}'):
                    if 'checkpointLM1_epoch' in ckpt:
                        os.remove(f'{best_model_dir}/{ckpt}') #avoid too many checkpoints
                logger.info('Saving best model : %s' % (best_model))
                torch.save(model, best_model)
                logger.info('Saving model checkpoint at : %s' % (saved_checkpoint_dir))
                model.save_pretrained(saved_checkpoint_dir)
                trigger_times = 0
                last_f1 = current_f1


        tmp = eval_results
        tmp['epoch'] =  epoch
        tmp['train_loss'] =  total_trn_loss
        final_results.append(tmp)
    df = pd.DataFrame(final_results)
    df.to_csv(f'{model_outputs}/log_v1_FUNSD_{test_mode}_run{run}.csv', index = False)
    best_model = torch.load(best_model)
    return best_model

def load_finetuned_model_or_base(checkpoint_path, base_model_name, num_labels):
    try:
        model = LayoutLMForTokenClassification.from_pretrained(checkpoint_path, num_labels=num_labels)
        logger.info("Loaded finetuned model from checkpoint: %s" %(checkpoint_path))
    except (OSError, ValueError):
        model = LayoutLMForTokenClassification.from_pretrained(base_model_name, num_labels=num_labels)
        logger.info("Loaded base pretrained model: %s" %(base_model_name))

    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=str, default=0)
    parser.add_argument("--adam_epsilon", type=str, default=1e-8)
    parser.add_argument("--weight_decay", type=str, default=0.0)
    parser.add_argument("--max_steps", type=str, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=str, default=1)
    parser.add_argument("--max_grad_norm", type=str, default=1.0)
    parser.add_argument("--early_stop_patience", type=str, default=6)
    parser.add_argument("--test_mode_metric", type=str, default='f1_score')
    parser.add_argument("--checkpoint_model_path", type=str, default=None)
    # Data, model, and output directories
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args_pass, _ = parser.parse_known_args()

    EPOCHS = args_pass.epochs
    TRAIN_BATCH_SIZE = args_pass.train_batch_size
    VALID_BATCH_SIZE = args_pass.eval_batch_size
    LEARNING_RATE = float(args_pass.learning_rate)
    LR_SCHEDULER_TYPE = args_pass.lr_scheduler_type
    WARMUP_STEPS = int(args_pass.warmup_steps)
    ADAM_EPSILON = float(args_pass.adam_epsilon)
    WEIGHT_DECAY = float(args_pass.weight_decay)
    MAX_STEPS = int(args_pass.max_steps)
    GRADIENT_ACC_STEPS = int(args_pass.gradient_accumulation_steps)
    MAX_GRAD_NORM = float(args_pass.max_grad_norm)
    PATIENCE = int(args_pass.early_stop_patience)
    TEST_MODE_METRIC = str(args_pass.test_mode_metric)
    CHECKPOINT_MODEL_PATH = str(args_pass.checkpoint_model_path)

    logger.info("All passed parameters values: ")
    logger.info("EPOCHS: %s" %(EPOCHS))
    logger.info("TRAIN_BATCH_SIZE: %s" %(TRAIN_BATCH_SIZE))
    logger.info("VALID_BATCH_SIZE: %s" %(VALID_BATCH_SIZE))
    logger.info("LEARNING_RATE: %s" %(LEARNING_RATE))
    logger.info("WARMUP_STEPS: %s" %(WARMUP_STEPS))
    logger.info("ADAM_EPSILON: %s" %(ADAM_EPSILON))
    logger.info("WEIGHT_DECAY: %s" %(WEIGHT_DECAY))
    logger.info("MAX_STEPS: %s" %(MAX_STEPS))
    logger.info("GRADIENT_ACC_STEPS: %s" %(GRADIENT_ACC_STEPS))
    logger.info("MAX_GRAD_NORM: %s" %(MAX_GRAD_NORM))
    logger.info("PATIENCE: %s" %(PATIENCE))
    logger.info("TEST_MODE_METRIC: %s" %(TEST_MODE_METRIC))
    logger.info("CHECKPOINT_MODEL_PATH: %s" %(CHECKPOINT_MODEL_PATH))
    logger.info("INPUT_DATA_DIRECTORY: %s" %(args_pass.data_dir))
    logger.info("OUTPUT_DIRECTORY: %s" %(args_pass.output_dir))

    # setting device on GPU if available, else CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s' %(DEVICE))

    # os.makedirs(args_pass.data_dir,exist_ok=True)
    try:
        os.makedirs(args_pass.output_dir, exist_ok=True)
        os.makedirs(f'{args_pass.output_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{args_pass.output_dir}/model_outputs', exist_ok=True)
        os.makedirs(f'{args_pass.output_dir}/models', exist_ok=True)
    except Exception as e:
        logger.info("Failed to create sub directories in %s" %(args_pass.output_dir))

    SAVED_CHECKPOINT_DIR = os.path.join(args_pass.output_dir, 'checkpoints')
    MODEL_OUTPUTS = os.path.join(args_pass.output_dir, 'model_outputs')
    BEST_MODEL_PATH = os.path.join(args_pass.output_dir, 'models')

    logger.info("Start loading labels.txt file from directory %s" %(args_pass.data_dir))
    try:
        #Loading labels.txt file
        labels = get_labels(f"{args_pass.data_dir}/labels.txt")
        num_labels = len(labels)
        label_map = {i: label for i, label in enumerate(labels)}
        logger.info("Successfully Loaded labels.txt file from directory %s" %(args_pass.data_dir))
    except Exception as e:
        logger.info("Failed to load labels.txt file from directory %s" %(args_pass.data_dir))

    args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': args_pass.data_dir,
        'model_name_or_path':'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',}

    args = AttrDict(args)

    logger.info("Loading layoutlm tokenizer....")
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    logger.info("Loading layoutlm base model(microsoft/layoutlm-base-uncased) or checkpoint from directory %s" %(CHECKPOINT_MODEL_PATH))
    base_model = load_finetuned_model_or_base(checkpoint_path=CHECKPOINT_MODEL_PATH, base_model_name="microsoft/layoutlm-base-uncased", num_labels=num_labels)

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    try:
        logger.info("Loading Dataset and creating dataloaders....")
        # the LayoutLM authors already defined a specific FunsdDataset, so we are going to use this here
        train_dataset = FunsdDataset(args=args, tokenizer=tokenizer, labels=labels, pad_token_label_id=pad_token_label_id, mode="train")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=TRAIN_BATCH_SIZE, num_workers=2)

        eval_dataset = FunsdDataset(args=args, tokenizer=tokenizer, labels=labels, pad_token_label_id=pad_token_label_id, mode="test")
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                    batch_size=VALID_BATCH_SIZE, num_workers=2)

        logger.info("Successfully Loaded Dataset and created train/test dataloaders.")
    except Exception as e:
        logger.info("Failed to load dataset %s" %(e))


    logger.info("Setting up Training parameters")
    # Training Parameters
    adam_epsilon = ADAM_EPSILON
    weight_decay = WEIGHT_DECAY
    num_train_epochs = EPOCHS ## To fine-tune (adding drop out so that It can lead to overfit less)
    max_steps = MAX_STEPS
    gradient_accumulation_steps = GRADIENT_ACC_STEPS
    max_grad_norm = MAX_GRAD_NORM
    warmup_steps = WARMUP_STEPS
    seed = 42

    early_stop = {'patience': PATIENCE}
    learning_rate = LEARNING_RATE
    # test_mode_metric = 'val_loss'
    test_mode_metric = TEST_MODE_METRIC
    runs = 1
    # optimizer = AdamW(base_model.parameters(), lr=learning_rate)

    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = (
            max_steps
            // (len(train_dataloader) // gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // gradient_accumulation_steps
            * num_train_epochs
        )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in base_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in base_model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(
            optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
        )
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

    try:
        logger.info("Started model training....")
        #start model training
        model = train_layoutLM(base_model, epochs=num_train_epochs, dataloader_train=train_dataloader, dataloader_eval=eval_dataloader,
                  optimizer=optimizer, scheduler=scheduler, early_stop_arg=early_stop, run=runs, test_mode=test_mode_metric, seed=seed, saved_checkpoint_dir=SAVED_CHECKPOINT_DIR, model_outputs=MODEL_OUTPUTS, best_model_dir=BEST_MODEL_PATH, device=DEVICE)

        logger.info("Successfully completed model training....")
    except Exception as e:
        logger.info("Failed to train model %s" %(e))













