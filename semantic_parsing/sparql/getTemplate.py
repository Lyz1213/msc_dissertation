import json

def read_txt(path):
    datas = []
    with open(path) as f:
        lines = f.readlines()
        data = {}
        for line in lines:
            if line is not None:
                line = line.strip()
                if line.startswith('Q'):
                    data['query'] = line.split('Q:')[1].strip()
                elif line.startswith('SPARQL'):
                    data['sparql'] = line.split('SPARQL:')[1].strip()
                elif line.startswith('ALIST'):
                    data['alist'] = line.split('ALIST:')[1].strip()
                if 'query' in data and 'sparql' in data and 'alist' in data:
                    datas.append(data)
                    data = {}
    return datas

def modify_file(file_pair):
    datas = read_txt(file_pair[1])
    new_datas = []
    templates  = []
    with open(file_pair[0]) as fjson:
        data = json.load(fjson)
        txt_i = 0
        while(True):
            incom_data = datas[txt_i]
            if (txt_i % 1000 == 0):
                print(txt_i)
            json_i = 0
            while(json_i < len(data)):
                question = data[json_i]['question']
                NNQT = data[json_i]['NNQT_question']
                # if json_i == 3447 or json_i == 4613 or json_i ==5317:
                #     incom_data['template'] = data[json_i]['template']
                #     templates.append(data[json_i]['template'])
                #     new_datas.append(incom_data)
                #     json_i += 1
                #     txt_i += 1
                #     break
                # if json_i >= 6226 and json_i < 6260:
                #     print(question)
                question = question.strip() if question!=None else None
                NNQT = NNQT.strip() if NNQT!=None else None
                if question == incom_data['query'] or NNQT == incom_data['query']:
                    if incom_data['query'] == None:
                        print('wrong')
                    incom_data['template'] = data[json_i]['template']
                    templates.append(str(data[json_i]['template']))
                    new_datas.append(incom_data)
                    txt_i += 1
                    break
                else:
                    json_i += 1
                if json_i >= len(data):
                    print('didnt find ', txt_i)
                    txt_i += 1
                    break
            if txt_i >= len(datas):
                break
    return new_datas, templates

def save_txt(filename, datas):
    with open(filename, 'a') as fwrite:
        for data in datas:
            string = '\n Q: ' + data['query'] + '\n SPARQL: ' + data['sparql'] + '\n ALIST: ' + data['alist'] + '\n TEMPLATE: ' + str(data['template']) + '\n'
            fwrite.writelines(string)


if __name__ == '__main__':
    train_pair = ('./data/train.json', './data/train.txt', './data/train_.txt')
    test_pair = ('./data/test.json', './data/test.txt', './data/test_.txt')
    valid_pair = ('./data/test.json', './data/valid.txt', './data/valid_.txt')
    data_pairs = [test_pair, valid_pair, train_pair]
    templates_list = []
    for pair in data_pairs:
        print(pair)
        new_datas, templates = modify_file(pair)
        templates_list.extend(templates)
        save_txt(pair[2], new_datas)
    templates_set = set(templates_list)
    with open('./data/templates.txt', 'a') as f:
        for template in templates_set:
            f.writelines(str(template) + '\n')
