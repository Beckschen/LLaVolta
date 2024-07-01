import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, COLOR_CHOICES, YES_NO_CHOICES, NUMBER_CHOICES, SIZE_CHOICES
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from llava.eval.assignment_viz import assignment_viz
import math

CHOICE_MAPPING = {
    'number': NUMBER_CHOICES,
    'color': COLOR_CHOICES,
    'yesno': YES_NO_CHOICES,
    'size': SIZE_CHOICES
}
CALC_LATENCY=False

def find_index_by_key(list_of_dicts, key, query):
    for index, dictionary in enumerate(list_of_dicts):
        if dictionary[key] == query:
            return index
    return -1  # Return -1 if the key is not found in any dictionary


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)

class ICLCustomDataset(CustomDataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, icl_file):
        super().__init__(questions, image_folder, tokenizer, image_processor, model_config)
        self.ICL = [json.loads(q) for q in open(os.path.expanduser(icl_file), "r")] if icl_file != "none" else None
        
    def cvt_icl(self, icl,category):
        op = DEFAULT_IMAGE_TOKEN + '\n'
        op += "Question: " + icl['text'] + '\n'
        op += " ASSISTANT: " + icl['answer'] 
        if category == 'number':
            choice = ', '.join(NUMBER_CHOICES)
        elif category == 'color':
            choice = ', '.join(COLOR_CHOICES)
        elif category == 'yesno':
            choice = ', '.join(YES_NO_CHOICES)
        elif category == 'size':
            choice = ', '.join(SIZE_CHOICES)
        # op += "Choices: " + "[" + choice + "]" + '\n'
        
        return op
             
    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        if "how many" in qs.lower():
            category = 'number'
        elif "color" in qs.lower():
            category = 'color'
        elif "size" in qs.lower():
            category = 'size'
        else:
            category = 'yesno'
        choices = ",".join(CHOICE_MAPPING[category])
        # based on category find the icl_idx
        if self.ICL is not None:
            icl_idx = find_index_by_key(self.ICL, 'category', category)
            icl = self.cvt_icl(self.ICL[icl_idx],category)
            qs = icl + '\n' + qs
        # qs = qs + '\n' + "Choices: " + "[" + choices + "]" + '\n' 
        # qs += "Answer Choice:"
        
        # print(qs)
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # print(prompt)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_ip = [image]
        if self.ICL is not None:
            image_ip.append(Image.open(os.path.join(self.image_folder, self.ICL[icl_idx]['image'])).convert('RGB'))
        image_ip = image_ip[::-1]
        image_tensor = process_images(image_ip, self.image_processor, self.model_config)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, icl_file='none',icl=False,batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    if icl:
        dataset = ICLCustomDataset(questions, image_folder, tokenizer, image_processor, model_config,icl_file)
    else:
        dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader




def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.post_config(args)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args.icl_file,args.icl)
    IDX = 1
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # if IDX % 50 == 0 and CALC_LATENCY:
        #     latency = torch.mean(torch.tensor(model.latency)).item()
        #     print(f" Latency: {latency:.5f}ms")


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        IDX += 1
        # ans_file.flush()
    ans_file.close()
str2bool = lambda x: (str(x).lower() == 'true')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--grouping", type=str, default='none')
    parser.add_argument("--halfpool", type=str2bool, default='false')
    parser.add_argument("--icl", action="store_true")
    parser.add_argument("--icl-file", type=str, default="none")
    parser.add_argument("--cot-decoding", action="store_true")
    args = parser.parse_args()

    eval_model(args)
