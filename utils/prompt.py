import json
import os
import torch
from utils import features


def get_prompts_tensors(args, model, tokenizer, logger):
    dataset_root = f'{args.retrieved_path}/{args.dataset}'
    metric_fn = f'./data/{args.dataset}/{args.dataset}_metrics-{args.database.upper()}.json'

    with open(metric_fn, 'r') as f:
        metrics = json.load(f)
    logger.info(f'Loaded metrics from: {metric_fn}')
    logger.info(f'len(metrics): {len(metrics)}')

    prompts_dir = os.path.join('data', args.dataset, 'prompts/')

    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
        logger.info(f'Created directory: {prompts_dir}')

    prompt_tensors_dict = {}
    text_prompts_dict = {}
    tokenized_text_prompts_dict = {}
    for label_type in [args.prompt_name]:
        prompt_tensors_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_prompt_tensors.pth"
        text_prompts_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_text_prompts.pth"
        tokenized_text_prompts_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_tokenized_text_prompts.pth"

        if os.path.exists(tokenized_text_prompts_filename):
            logger.info(f'Loading prompt tensors from {prompt_tensors_filename}')
            prompt_tensors = torch.load(prompt_tensors_filename, weights_only=True)
            prompt_tensors_dict[label_type] = prompt_tensors

            text_prompts = torch.load(text_prompts_filename, weights_only=True)
            text_prompts_dict[label_type] = text_prompts

            tokenized_text_prompts = torch.load(tokenized_text_prompts_filename, weights_only=True)
            tokenized_text_prompts_dict[label_type] = tokenized_text_prompts

        else:
            logger.info(f'Calculating prompt tensors for {label_type} ...')
            text_prompts = prompt_maker(metrics, args.dataset, label_type)
            text_prompts_dict[label_type] = text_prompts
            torch.save(text_prompts, text_prompts_filename)
            logger.info(f'Saved text prompts to {text_prompts_filename}')

            # tokenize the text_prompts first in case of finetune needed
            tokenized_text_prompts = features.get_text_features(model, text_prompts, tokenize=tokenizer,
                                                                operation='tokenize')
            tokenized_text_prompts_dict[label_type] = tokenized_text_prompts
            torch.save(tokenized_text_prompts, tokenized_text_prompts_filename)
            logger.info(f'Saved tokenized text prompts to {tokenized_text_prompts_filename}')

            prompt_tensors = features.get_text_features(model, text_prompts, tokenizer, 'encode')
            prompt_tensors_dict[label_type] = prompt_tensors
            torch.save(prompt_tensors, prompt_tensors_filename)
            logger.info(f'Saved prompt tensors to {prompt_tensors_filename}')

    return prompt_tensors_dict, text_prompts_dict, tokenized_text_prompts_dict


def set_prompt(args, model, tokenizer, logger):
    prompt_tensors_dict, text_prompts_dict, \
        tokenized_text_prompts_dict = get_prompts_tensors(args, model, tokenizer, logger)

    prompt_tensors = prompt_tensors_dict[args.prompt_name]
    text_prompts = text_prompts_dict[args.prompt_name]
    tokenized_text_prompts = tokenized_text_prompts_dict[args.prompt_name]

    return prompt_tensors, text_prompts, tokenized_text_prompts, prompt_tensors_dict


def prompt_maker(metrics: dict, dataset_name: str, name_type='most_common_name'):
    prompts = {}
    if dataset_name == 'semi_aves':
        prompt_templates = TEMPLATES_DIC[dataset_name][name_type]
    else:
        prompt_templates = TEMPLATES_DIC[dataset_name]
    # print('prompt_templates:', prompt_templates)

    for i, key in enumerate(metrics.keys()):
        label = metrics[key][name_type]

        if name_type == 'alternates':
            prompt_lst = []
            for alt_name, ct in label.items():
                pt = [template.format(alt_name) for template in prompt_templates]
                prompt_lst.extend(pt)
            prompts[key] = {'corpus': prompt_lst}
        else:
            prompts[key] = {'corpus': [template.format(label) for template in prompt_templates]}

    prompts = dict(sorted(prompts.items(), key=lambda x: int(x[0])))

    return prompts


def prompt_maker_aves(metrics: dict, dataset_name: str, name_type='s-name'):
    prompts = {}
    prompt_templates = TEMPLATES_DIC[dataset_name][name_type.split('_')[0]]
    print('prompt_templates:', prompt_templates)

    for i, key in enumerate(metrics.keys()):
        class_id = str(metrics[key]['class_id'])
        s_name = metrics[key]['species']
        c_name = metrics[key]['common_name']

        if name_type == 's-name':
            prompt_lst = [template.format(s_name) for template in prompt_templates]

        elif name_type == 'c-name':
            prompt_lst = [template.format(c_name) for template in prompt_templates]

        elif name_type == 't-name':
            prompt_lst = [template.format(s_name, c_name) for template in prompt_templates]

        elif name_type == 'f-name':
            freq_name = metrics[key]['most_freq_synonym']
            prompt_lst = [template.format(freq_name) for template in prompt_templates]

        elif name_type == 'c-name-80prompts':
            prompt_lst = [template.format(c_name) for template in prompt_templates]

        elif name_type == 'c-name_attribute':
            prompt_lst = [template.format(c_name) for template in prompt_templates]
            attributes = json.load(open('data/semi_aves/prompts/visual-attrs-semi_aves.json', 'r'))
            attributes_lst = attributes[key]["corpus"]
            attributes_prompt_lst = [template.format(c_name) + f' {c_name} {attr}'.replace('Has', 'has') for template in
                                     prompt_templates for attr in attributes_lst]
            prompt_lst.extend(attributes_prompt_lst)

        prompts[class_id] = {'corpus': prompt_lst}

    prompts = dict(sorted(prompts.items(), key=lambda x: int(x[0])))

    return prompts

fgvcaircraft_templates = [
    'a photo of a {}, a type of aircraft.',
    'a photo of the {}, a type of aircraft.',
]

food101_templates = [
    'a photo of {}, a type of food.',
]

oxfordpets_templates = [
    'a photo of a {}, a type of pet.',
]

stanfordcars_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a photo of my {}.',
    'i love my {}!',
    'a photo of my dirty {}.',
    'a photo of my clean {}.',
    'a photo of my new {}.',
    'a photo of my old {}.',
]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

semi_aves_templates = {
    's-name': ['a photo of a {}, a type of bird.'],
    'c-name': ['a photo of a {}, a type of bird.'],
    't-name': ['a photo of a {}, a type of bird, commonally known as {}.'],
    'f-name': ['a photo of a {}, a type of bird.'],
    'most_common_name': ['a photo of a {}, a type of bird.'],
    'most_common_name_REAL': ['a photo of a {}, a type of bird.'],
    'name': ['a photo of a {}, a type of bird.'],
    'c-name-80prompts': imagenet_templates,
}

TEMPLATES_DIC = {
    'food101': food101_templates,
    'stanford_cars': stanfordcars_templates,
    'aircraft': fgvcaircraft_templates,
    'oxford_pets': oxfordpets_templates,
    'semi_aves': semi_aves_templates,
}

