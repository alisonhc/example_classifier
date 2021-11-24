import os
import random
import jsonlines


def check_if_usable_sentence(txt):
    punc = {'.', ',', ';', '!', '?'}
    quotes = {"'", '"'}
    if txt and len(txt) > 3 and txt[0].isupper() and ((txt[-1] in punc) or (txt[-2] in punc) and txt[-1] in quotes):
        return True
    return False


def recursively_expand_slashes(toks):
    example_list = []

    def recurse_slashes(tokens, new_tokens):
        if not tokens:
            example_list.append(' '.join(new_tokens))
            return
        to_check = tokens[0]
        if '/' in to_check:
            options = to_check.split('/')
            for opt in options:
                recurse_slashes(tokens[1:], new_tokens + [opt])
        else:
            recurse_slashes(tokens[1:], new_tokens + [to_check])
    recurse_slashes(toks, [])
    return example_list


def create_jsonl_evp_data(evp_path, sentences=True):
    data_list = []
    with open(evp_path) as evp:
        lines = evp.readlines()
        for line in lines:
            info = line.split('\t')
            examples_to_consider = []
            if len(info) >= 7:
                label = info[4].strip().upper()
                dict_example = info[6].strip()
                dict_example = {'text': dict_example, 'source': 'dict'}
                examples_to_consider.append(dict_example)
            if len(info) == 8:
                student_example = info[7].strip()
                student_example = {'text': student_example, 'source': 'student'}
                examples_to_consider.append(student_example)
            for example in examples_to_consider:
                if sentences:
                    if check_if_usable_sentence(example['text']):
                        data_list.append({"text": example['text'], "label": label, 'source': example['source']})
                else:
                    toks = example['text'].split()
                    tok_len = len(toks)
                    if not check_if_usable_sentence(example['text']) and 6 > tok_len > 1:
                        for phrase in recursively_expand_slashes(toks):
                            data_list.append({"text": phrase, "label": label, 'source': example['source']})
    print(len(data_list))
    with jsonlines.open(os.path.join('data', 'evp_phrases_1.jsonl'), 'w') as writer:
        for sample in data_list:
            writer.write(sample)


def create_train_test_dev_data(fpath, start='evp'):
    lines = []
    with jsonlines.open(fpath) as reader:
        for obj in reader:
            lines.append(obj)
    train_num = int(len(lines)*0.8)
    random.shuffle(lines, seed=42)
    train_samples = lines[:train_num]
    test_dev_samples = lines[train_num:]
    test_num = int(len(test_dev_samples)*0.5)
    test_samples = test_dev_samples[:test_num]
    dev_samples = test_dev_samples[test_num:]
    for tup in [('train', train_samples), ('test', test_samples), ('dev', dev_samples)]:
        print(f"{tup[0]} samples: {len(tup[1])}")
        if tup[0] != 'test':
            with jsonlines.open(os.path.join('..', 'data', f'{start}_{tup[0]}.jsonl'), 'w') as writer:
                for sample in tup[1]:
                    writer.write(sample)
        else:
            dict_examples = []
            student_examples = []
            for sample in tup[1]:
                if sample['source'] == 'dict':
                    dict_examples.append(sample)
                else:
                    student_examples.append(sample)
            for src in [('dict', dict_examples), ('student', student_examples)]:
                with jsonlines.open(os.path.join('..', 'data', f'{start}_{tup[0]}_{src[0]}.jsonl'), 'w') as w:
                    for s in src[1]:
                        w.write(s)


def retroactively_split_test_data(dataset_path, test_path):
    evp_dict = {}
    with open(dataset_path) as evp:
        lines = evp.readlines()
        for line in lines:
            info = line.split('\t')
            if len(info) >= 7:
                dict_example = info[6].strip()
                evp_dict[dict_example] = 'dict'
            if len(info) == 8:
                student_example = info[7].strip()
                evp_dict[student_example] = 'student'
    dict_examples = []
    student_examples = []
    with jsonlines.open(test_path) as reader:
        for obj in reader:
            if evp_dict[obj['text']] == 'dict':
                dict_examples.append(obj)
            elif evp_dict[obj['text']] == 'student':
                student_examples.append(obj)
    print(len(dict_examples), len(student_examples))
    for src in [('dict', dict_examples), ('student', student_examples)]:
        with jsonlines.open(os.path.join('data', f'evp_test_{src[0]}.jsonl'), 'w') as w:
            for s in src[1]:
                w.write(s)

def convert_txt_to_jsonl(fpath):
    obj_list = []
    with open(fpath) as reader:
        lines = reader.readlines()
        for l in lines:
            info = l.split('\t')
            text = info[0].strip()
            label = info[1].strip()
            obj_list.append({'text': text, 'label': label})
    with jsonlines.open(os.path.join('data', f'voa_test.jsonl'), 'w') as w:
        for obj in obj_list:
            w.write(obj)
    print(len(obj_list))



if __name__ == '__main__':
    evp_path = ''

