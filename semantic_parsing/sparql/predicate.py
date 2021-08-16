import numpy as np
import re
import json
from wikidata.client import Client
data_path = './data'
entity_path = './data/resource/entities_covered'
predicate_path = './data/resource/predicates_with_frequency'
predictae_dict = {}
client = Client()
wf = open('./data/predicate_dict.txt','a')
mark = 'P653'
count = 0
with open(predicate_path) as f:
    lines = f.readlines()
    flag = False
    for line in lines:
        if line.__contains__('P'):
            line = line.strip().strip().split(": ")
            line[0] = re.sub(r'[^\w\s]', '', line[0])
            print(line[0])
            print(line[1])
            if flag:
                plist = []
                predicate = client.get(line[0], load=True)
                label =predicate.label
                plist.append(str(label))
                print("{} {}".format(predicate, label))
                if 'aliases' in predicate.data:
                    if 'en' in predicate.data['aliases']:
                        for dict in predicate.data['aliases']['en']:
                            plist.append(dict['value'])
                string = str(line[0]) + " : " + str(plist) + "\n"
                wf.writelines(string)
            else:
                count += 1
            if line[0] == mark:
                print('count is ', count)
                flag = True
