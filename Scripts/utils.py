import torch
from torch.utils.data import Dataset
import collections
from transformers import LayoutLMTokenizer
import logging, os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_labels(path):
    try:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        # if "O" not in labels:
        #     labels = ["O"] + labels
        d = {}
        for x in labels:
            d[x] = 1
        final_labels = list(d.keys())
        return final_labels
    except Exception as e:
        logger.info("Exception raised in get_labels: %s" % (e))


class FunsdDataset(Dataset):
    # global count
    def __init__(self, args, tokenizer, labels, pad_token_label_id, mode):
        if args.local_rank not in [-1, 0] and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s" %(cached_features_file))
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s" %(args.data_dir))
            examples = read_examples_from_file(args.data_dir, mode)
            features, _, _, _, _, _ = convert_examples_to_features(
                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(args.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=pad_token_label_id
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s" %(cached_features_file))
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)
        # for f in filename:
        #     if f.file_name not in image_path_dict:
        #         count = count + 1
        #         image_path_dict[count] = f.file_name
        self.all_filename = [f.file_name for f in features]
        # count = count + 1

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
            self.all_filename[index],
        )


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        boxes,
        actual_bboxes,
        file_name,
        page_size,
    ):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


def read_examples_from_file(data_dir, mode):
    try:
        file_path = os.path.join(data_dir, "{}.txt".format(mode))
        box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
        image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f, open(
            box_file_path, encoding="utf-8"
        ) as fb, open(image_file_path, encoding="utf-8") as fi:
            words = []
            boxes = []
            actual_bboxes = []
            file_name = None
            page_size = None
            labels = []
            for idx, (line, bline, iline) in enumerate(zip(f, fb, fi)):
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(
                            InputExample(
                                guid="{}-{}".format(mode, guid_index),
                                words=words,
                                labels=labels,
                                boxes=boxes,
                                actual_bboxes=actual_bboxes,
                                file_name=file_name,
                                page_size=page_size,
                            )
                        )
                        guid_index += 1
                        words = []
                        boxes = []
                        actual_bboxes = []
                        file_name = None
                        page_size = None
                        labels = []
                else:
                    splits = line.split("\t")
                    bsplits = bline.split("\t")
                    isplits = iline.split("\t")
                    try:
                        assert splits[0] == bsplits[0]
                        assert len(splits) == 2
                        assert len(bsplits) == 2
                        assert len(isplits) == 4
                    except:
                        print(idx, splits[0], "-----", bsplits[0], isplits[-1])

                    words.append(splits[0])
                    if len(splits) > 1:
                        if splits[-1].replace("\n", "") == "" or splits[-1].replace("\n", "") == 0:
                          print("wrong labels")
                        labels.append(splits[-1].replace("\n", ""))
                        box = bsplits[-1].replace("\n", "")
                        box = [int(b) for b in box.split()]
                        boxes.append(box)
                        actual_bbox = [int(b) for b in isplits[1].split()]
                        actual_bboxes.append(actual_bbox)
                        page_size = [int(i) for i in isplits[2].split()]
                        file_name = isplits[3].strip()
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(
                    InputExample(
                        guid="%s-%d".format(mode, guid_index),
                        words=words,
                        labels=labels,
                        boxes=boxes,
                        actual_bboxes=actual_bboxes,
                        file_name=file_name,
                        page_size=page_size,
                    )
                )
        return examples
    except Exception as e:
        logger.info("Exception raised in read_examples_from_file: %s" % (e))


def _check_is_max_context(doc_spans, cur_span_index, position):
    try:
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index
    except Exception as e:
        logger.info("Exception raised in _check_is_max_context: %s" % (e))


original_token = []
def convert_single_example_to_feature(
            ex_index,
            example,
            label_list,
            max_seq_length,
            tokenizer,
            cls_token_at_end=False,
            cls_token="[CLS]",
            cls_token_segment_id=1,
            sep_token="[SEP]",
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=0,
            cls_token_box=[0, 0, 0, 0],
            sep_token_box=[1000, 1000, 1000, 1000],
            pad_token_box=[0, 0, 0, 0],
            pad_token_segment_id=0,
            pad_token_label_id=-100,
            sequence_a_segment_id=0,
            mask_padding_with_zero=True,
            doc_stride=384
        ):
        try:
            """ Loads a data file into a list of `InputBatch`s
                `cls_token_at_end` define the location of the CLS token:
                    - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                    - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
                `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
            """
            # label_map = {label: i for i, label in enumerate(label_list)}
            file_name = example.file_name
            page_size = example.page_size
            width, height = page_size

            label_map = {label: i for i, label in enumerate(label_list)}

            tokens = []
            token_boxes = []
            actual_bboxes = []
            label_ids = []
            for word, label, box, actual_bbox in zip(
                example.words, example.labels, example.boxes, example.actual_bboxes
            ):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                token_boxes.extend([box] * len(word_tokens))
                actual_bboxes.extend([actual_bbox] * len(word_tokens))
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend(
                    [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
                )

            # print(label_ids, len(label_ids))
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2

            # print(">>>>>>>>>>>>>>>>", len(tokens))
            max_tokens_for_doc = max_seq_length - 2
            # max_tokens_for_doc = max_seq_length - len(tokens) - 2
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(tokens):
                length = len(tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(tokens):
                    break
                start_offset += min(length, doc_stride)

            feature_list, ntokens_list, label_ids_list, actual_boxes_list, token_boxes_list = [], [], [], [], []

            for (doc_span_index, doc_span) in enumerate(doc_spans):

                token_is_max_context = {}
                token = []
                token_box = []
                actual_bbox = []
                label_id = []
                segment_id = []
                # print(doc_span.length, doc_spans)
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    # token_to_orig_map[len(ntokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                          split_token_index)
                    token_is_max_context[split_token_index] = is_max_context
                    token.append(tokens[split_token_index])
                    token_box.append(token_boxes[split_token_index])
                    actual_bbox.append(actual_bboxes[split_token_index])
                    label_id.append(label_ids[split_token_index])
                    # label_id.append(label_map[label_ids[split_token_index]])
                    segment_id.append(0)

                # print(label_id)
                token += [sep_token]
                token_box += [sep_token_box]
                actual_bbox += [[0, 0, width, height]]
                label_id += [pad_token_label_id]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    token += [sep_token]
                    token_box += [sep_token_box]
                    actual_bbox += [[0, 0, width, height]]
                    label_id += [pad_token_label_id]
                segment_id = [sequence_a_segment_id] * len(token)

                if cls_token_at_end:
                    token += [cls_token]
                    token_box += [cls_token_box]
                    actual_bbox += [[0, 0, width, height]]
                    label_id += [pad_token_label_id]
                    segment_id += [cls_token_segment_id]
                else:
                    token = [cls_token] + token
                    token_box = [cls_token_box] + token_box
                    actual_bbox = [[0, 0, width, height]] + actual_bbox
                    label_id = [pad_token_label_id] + label_id
                    segment_id = [cls_token_segment_id] + segment_id

                input_id = tokenizer.convert_tokens_to_ids(token)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_id)
                if pad_on_left:
                    input_id = ([pad_token] * padding_length) + input_id
                    input_mask = (
                        [0 if mask_padding_with_zero else 1] * padding_length
                    ) + input_mask
                    segment_id = ([pad_token_segment_id] * padding_length) + segment_id
                    label_id = ([pad_token_label_id] * padding_length) + label_id
                    token_box = ([pad_token_box] * padding_length) + token_box
                else:
                    input_id += [pad_token] * padding_length
                    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                    segment_id += [pad_token_segment_id] * padding_length
                    label_id += [pad_token_label_id] * padding_length
                    token_box += [pad_token_box] * padding_length

                # print(len(input_id), len(input_mask), len(segment_id), len(label_id), len(token_box))
                assert len(input_id) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_id) == max_seq_length
                assert len(label_id) == max_seq_length
                assert len(token_box) == max_seq_length
                # assert len(token) == max_seq_length

                feature = InputFeatures(
                        input_ids=input_id,
                        input_mask=input_mask,
                        segment_ids=segment_id,
                        label_ids=label_id,
                        boxes=token_box,
                        actual_bboxes=actual_bbox,
                        file_name=file_name,
                        page_size=page_size,
                    )
                feature_list.append(feature)
                ntokens_list.append(token)
                label_ids_list.append(label_id)
                actual_boxes_list.append(actual_bbox)
                token_boxes_list.append(token_box)

            if ex_index < 2:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" %(example.guid))
                    logger.info("size of tokens: %s" %(len([str(x) for x in token])))
                    logger.info("size of input_ids: %s" %(len([str(x) for x in input_id])))
                    logger.info("size of input_mask: %s" %(len([str(x) for x in input_mask])))
                    logger.info("size of segment_ids: %s" %(len([str(x) for x in segment_id])))
                    logger.info("size of label_ids: %s" %(len([str(x) for x in label_id])))
                    logger.info("size of boxes: %s" %(len([str(x) for x in token_box])))
                    logger.info("size of actual_bboxes: %s" %(len([str(x) for x in actual_bbox])))
            return feature_list, ntokens_list, label_ids_list, actual_boxes_list, token_boxes_list
        except Exception as e:
            logger.info("Exception raised in convert_single_example_to_feature: %s" % (e))



def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    doc_stride=384
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    try:
        batch_tokens = []
        batch_labels = []
        batch_tokens_boxes = []
        batch_actual_bboxes = []
        batch_index = []
        feature_list_total = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logging.info("Writing example %d of %d" % (ex_index, len(examples)))
            feature_list, ntokens_list, label_ids_list, actual_boxes_list, token_boxes_list = convert_single_example_to_feature(ex_index, example, label_list, max_seq_length, tokenizer,
                cls_token_at_end=False,
                cls_token="[CLS]",
                cls_token_segment_id=1,
                sep_token="[SEP]",
                sep_token_extra=False,
                pad_on_left=False,
                pad_token=0,
                cls_token_box=[0, 0, 0, 0],
                sep_token_box=[1000, 1000, 1000, 1000],
                pad_token_box=[0, 0, 0, 0],
                pad_token_segment_id=0,
                pad_token_label_id=-100,
                sequence_a_segment_id=0,
                mask_padding_with_zero=True,
                doc_stride = 384)
            feature_list_total.extend(feature_list)
            for feature, ntokens, label_ids, actual_boxes, token_boxes in zip(feature_list, ntokens_list, label_ids_list, actual_boxes_list, token_boxes_list):
                batch_tokens.extend(ntokens)
                batch_labels.extend(label_ids)
                batch_index.extend([ex_index]*len(ntokens))
                batch_tokens_boxes.extend(token_boxes)
                batch_actual_bboxes.extend(actual_boxes)

        return feature_list_total, batch_tokens, batch_labels, batch_index, batch_tokens_boxes, batch_actual_bboxes
    except Exception as e:
        logger.info("Exception raised in convert_examples_to_features: %s" % (e))
