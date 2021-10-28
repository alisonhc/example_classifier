import os
import random
import jsonlines


def check_if_usable_sentence(txt):
    punc = {'.', ',', ';', '!', '?'}
    if txt and len(txt) > 3 and txt[0].isupper() and txt[-1] in punc:
        return True
    return False


def create_jsonl_evp_data(evp_path):
    data_list = []
    with open(evp_path) as evp:
        lines = evp.readlines()
        for line in lines:
            info = line.split('\t')
            examples_to_consider = []
            if len(info) >= 7:
                label = info[4].strip().upper()
                dict_example = info[6].strip()
                examples_to_consider.append(dict_example)
            if len(info) == 8:
                student_example = info[7].strip()
                examples_to_consider.append(student_example)
            for example in examples_to_consider:
                if check_if_usable_sentence(example):
                    data_list.append({"text": example, "label": label})
    print(len(data_list))
    with jsonlines.open(os.path.join('data', 'evp_all.jsonl'), 'w') as writer:
        for sample in data_list:
            writer.write(sample)


def create_train_test_dev_data(fpath, start='evp'):
    lines = []
    with jsonlines.open(fpath) as reader:
        for obj in reader:
            lines.append(obj)
    train_num = int(len(lines)*0.8)
    random.shuffle(lines)
    train_samples = lines[:train_num]
    test_dev_samples = lines[train_num:]
    test_num = int(len(test_dev_samples)*0.5)
    test_samples = test_dev_samples[:test_num]
    dev_samples = test_dev_samples[test_num:]
    for tup in [('train', train_samples), ('test', test_samples), ('dev', dev_samples)]:
        print(f"{tup[0]} samples: {len(tup[1])}")
        with jsonlines.open(os.path.join('data', f'{start}_{tup[0]}.jsonl'), 'w') as writer:
            for sample in tup[1]:
                writer.write(sample)
