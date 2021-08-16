import json
import re
import ast
from wikidata.client import Client

prediatce_path = './data/predicate_dict.txt'
test_path = './data/all_train.txt'
def split_sparql(sparql):
    sparql = sparql.strip().replace("}","").split("{")
    sparql[0] = sparql[0].strip()
    sparql[1] = sparql[1].strip().split(" . ")
    for i in range(len(sparql[1])):
        sparql[1][i] = sparql[1][i].strip()
    return sparql

def parse_where(whereclause, pv, question, qqnt, pre_dict, client):
    av = {}
    psudo_alist = []
    nnqt = False
    NNQT = False
    for triple in whereclause:
        if triple.lower().__contains__('filter'):
            if triple.lower().__contains__('year'):
                triple = triple.strip().split()
                if len(triple) == 4:
                    t_list = []
                    time_filter = triple[3].lower()
                    v_pattern = re.compile(r'year\((.*?)[)]', re.S)
                    variable = re.findall(v_pattern, time_filter)
                    t_pattern = re.compile(r'[\'](.*?)[\']', re.S)
                    time = re.findall(t_pattern, time_filter)
                    print("v is {} and time is {}".format(variable[0], time[0]))
                    t_list.append("t:"+str(time[0]))
                    alist_triple, av, NNQT = standardize(t_list, av, pv, pre_dict, question, qqnt)
                    psudo_alist.append(tuple(alist_triple))
                else:
                    print('do not follow the common pattern, the triple is {}'.format(triple))
            elif triple.lower().__contains__('contains'):
                triple = triple.strip().split()
                if len(triple) == 4:
                    # example : When was Trento's population at 117185
                    # SELECT ?value WHERE { wd:Q3376 p:P1082 ?s . ?s ps:P1082 ?x filter(contains(?x,'117185')) . ?s pq:P585 ?value}
                    contain_filter = triple[3].lower()
                    print('contain_filter', contain_filter)
                    v_pattern = re.compile(r'\?(.*?)[\,]', re.S)
                    variable = re.findall(v_pattern, contain_filter)
                    variable = "?" + str(variable[0])
                    t_pattern = re.compile(r'[\'](.*?)[\']', re.S)
                    term = re.findall(t_pattern, contain_filter)
                    print("v is {} and time is {}".format(variable, term[0]))
                    if variable == triple[2]:
                        triple[2] = "VALUE:" + term[0]
                        alist_triple, av, NNQT = standardize(triple[:-1], av, pv, pre_dict, question, qqnt)
                        psudo_alist.append(tuple(alist_triple))
                elif len(triple) == 2:
                    contain_filter = triple[0].lower()
                    obj = triple[1]
                    v_pattern = re.compile(r'\?(.*?)[\)]', re.S)
                    variable = re.findall(v_pattern, contain_filter)
                    variable = "?" + str(variable[0])
                    t_pattern = re.compile(r'[\'](.*?)[\']', re.S)
                    term = re.findall(t_pattern, obj)
                    c_list = []
                    c_list.append(variable)
                    c_list.append("VALUE:contains")
                    c_list.append("VALUE:"+term[0])
                    alist_triple, av, NNQT = standardize(c_list, av, pv, pre_dict, question, qqnt)
                    psudo_alist.append(tuple(alist_triple))

                else:
                    print('do not follow the common pattern, the triple is {}'.format(triple))
            elif triple.lower().__contains__('strstarts'):
                v_pattern = re.compile(r'\?(.*?)[\)]', re.S)
                variable = re.findall(v_pattern, triple)
                variable = "?" + str(variable[0])
                t_pattern = re.compile(r'[\'](.*?)[\']', re.S)
                term = re.findall(t_pattern, triple)
                s_list = [variable, 'VALUE:starts_with', 'VALUE:'+str(term[0])]
                alist_triple, av, NNQT = standardize(s_list, av, pv, pre_dict, question, qqnt)
                psudo_alist.append(tuple(alist_triple))
            else:
                if not triple.lower().__contains__('lang'):
                    operator = None
                    h = None
                    if triple.__contains__('='):
                        operator = '='
                        h = 'EQUAL'
                    elif triple.__contains__('<'):
                        operator = '<'
                        h = 'LESS'
                    elif triple.__contains__('>'):
                        operator = '>'
                        h = 'GREATER'
                    else:
                        print('do not follow the common pattern, the triple is {}'.format(triple))
                    if triple.__contains__('FILTER'):
                        triple = triple.strip().split('FILTER')
                    else:
                        triple = triple.strip().split('filter')
                    alist_triple, av, NNQT = standardize(triple[0].strip().split(), av, pv, pre_dict, question, qqnt)
                    psudo_alist.append(tuple(alist_triple))
                    eqfilter = triple[1].strip('(').strip(')').split(operator)
                    eq_list = [str(eqfilter[0]).strip(), 'OPERATOR:' + h + " " + str(eqfilter[1]).strip()]
                    alist_triple, av, NNQT = standardize(eq_list, av, pv, pre_dict, question, qqnt)
                    psudo_alist.append(tuple(alist_triple))

        elif triple.lower().__contains__('order'):
            print('order by: triple is {}'.format(triple))
            if triple.__contains__(':P'):
                split_triple = triple.strip().split()
                half_triple = split_triple[:3]
                alist_triple, av, NNQT = standardize(half_triple, av, pv, pre_dict, question, qqnt)
                psudo_alist.append(tuple(alist_triple))
                triple = ""
                for i in range(4, len(split_triple)):
                    triple += str(split_triple[i]) + " "
                triple = triple.strip()
            v_pattern = re.compile(r'\?(.*?)[\)]', re.S)
            variable = re.findall(v_pattern, triple)
            variable = "?" + str(variable[0])
            if triple.lower().__contains__('asc'):
                operator = "VALUE:MIN"
            elif triple.lower().__contains__('desc'):
                operator = "VALUE:MAX"
            o_list = []
            o_list.append(operator)
            o_list.append(variable)
            alist_triple, av, NNQT = standardize(o_list, av, pv, pre_dict, question, qqnt)
            psudo_alist.append(tuple(alist_triple))
        else:
            triple = triple.strip().split()
            if len(triple) != 3:
                print('something whent wrong, the triple is ', triple)
                if len(triple) == 4:
                    if triple[3] == '.':
                        alist_triple, av, NNQT = standardize(triple[:-1], av, pv, pre_dict, question, qqnt)
                        psudo_alist.append(tuple(alist_triple))
            else:
                alist_triple, av, NNQT = standardize(triple, av, pv, pre_dict, question, qqnt)
                psudo_alist.append(tuple(alist_triple))
        if NNQT:
            nnqt = True
    print('psudo_alist is {}'.format(psudo_alist))
    return psudo_alist, nnqt

def standardize(triple, av, pv, pre_dict, question, qqnt):
    alist_triple = []
    NNQT = False
    for term in triple:
        if term.__contains__('?'):
            if term in pv:
                pro_va = '?x' + str(pv.index(term))
                alist_triple.append(pro_va)
            else:
                if term in av:
                    alist_triple.append(av[term])
                else:
                    if len(pv) == 0:
                        aux_va = '?x' + str(len(av))
                    else:
                        aux_va = '$y' + str(len(av))
                    av[term] = aux_va
                    alist_triple.append(aux_va)
        else:
            # print('term is ', term)
            pair = term.split(':')
            att = pair[0]
            ID = pair[1]
            if att == 'VALUE':
                alist_triple.append(ID)
            elif att == 't':
                alist_triple.append(term)
            elif att == 'OPERATOR':
                ID = ID.split()
                alist_triple.append(ID[0])
                alist_triple.append(ID[1])
            elif ID == 'label':
                alist_triple.append(ID)
            elif ID == 'P31':
                alist_triple.append('type')
            elif ID == 'P585':
                alist_triple.append('TIME')
            elif ID == 'P706' or ID == 'P276':
                alist_triple.append('location')
            else:
                labels = fromWiki(ID, pre_dict, client)
                # print("ID is {} and label is {}".format(ID,labels))
                contains = False
                qqnt_contains = True
                for term in labels:
                    if question.__contains__(term):
                        contains = True
                        alist_triple.append(term)
                        break
                if not contains:
                    qqnt_contains = False
                    for term in labels:
                        print(labels)
                        if qqnt.__contains__(term):
                            NNQT = True
                            qqnt_contains = True
                            alist_triple.append(term)
                            break
                if not qqnt_contains:
                    return ['bad', 'data'], av, NNQT
                    #print('ID is {} and label is {}'.format(ID, labels))
    return alist_triple, av, NNQT

def load_predicate(path):
    predicate_dict = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' : ')
            pre_list = ast.literal_eval(line[1])
            predicate_dict[line[0]] = pre_list
    return predicate_dict

def fromWiki(ID, pre_dict, client):
    data = []
    if ID not in pre_dict:
        if ID.__contains__('Q') or ID.__contains__('P'):
            property = client.get(ID, load=True)
            data.append(str(property.label))
            if 'aliases' in property.data:
                if 'en' in property.data['aliases']:
                    for dict in property.data['aliases']['en']:
                        data.append(dict['value'])
        else:
            data.append(ID)
        return data
    else:
        return pre_dict[ID]

def check_Operator(select):
    select = select.lower()
    if select.__contains__('distinct'):
        return 'List'
    elif select.__contains__('count'):
        return 'Count'
    elif select.__contains__('ask'):
        return 'Is'
    else:
        return 'Value'

def trans2alist(operator, psudo_alist, pv):
    spv = ""
    for i in range(len(pv)):
        if i == 0:
            spv += '?x' + str(pv.index(pv[i]))
        else:
            spv += ',?x' + str(pv.index(pv[i]))
    vs = []
    v_triple = {}
    triple_v = {}
    pre_triple = None
    deleted_va = None
    changed_psudo = []
    change = True
    intersect = False
    for i in range(len(psudo_alist)):
        triple = psudo_alist[i]
        changed_psudo.append(triple)
        if pre_triple == None and len(triple) == 3:
            pre_triple = triple
        else:
            if len(triple) == 3:
                if triple[0] == pre_triple[2] and triple[1] == pre_triple[1]:
                    combined = (pre_triple[0], pre_triple[1], triple[2])
                    changed_psudo[i-1] = combined
                    changed_psudo.remove(triple)
                    deleted_va = triple[0]
                else:
                    if triple[0] == deleted_va or triple[2] == deleted_va:
                        change = False
                        break
                    pre_triple = triple
    if change:
        psudo_alist = changed_psudo

    for triple in psudo_alist:
        triple_v[triple] = []
        for term in triple:
            if term.__contains__('?') or term.__contains__('$'):
                triple_v[triple].append(term)
                if term not in vs:
                    vs.append(term)
    for triple, va in triple_v.items():
        t = None
        maxmin = None
        special_va = None
        compare = None
        if len(va) == 0:
            if str(triple).__contains__('t:'):
                t = triple[0].split(':')[1]
            else:
                if 'IS' not in v_triple:
                    v_triple['IS'] = []
                v_triple['IS'].append(triple)
        elif len(va) == 1:
            if triple[0].__contains__('MIN') or triple[0].__contains__('MAX'):
                maxmin = triple[0]
                special_va = va[0]
            elif triple[1].__contains__('EQUAL') or triple[1].__contains__('GREATER') or triple[1].__contains__('LESS'):
                operator = triple[1]
                compare = triple[2]
            elif triple[1].__contains__('TIME'):
                t = triple[2]
                spv = triple[2]
            else:
                if va[0] not in v_triple:
                    v_triple[va[0]] = []
                v_triple[va[0]].append(triple)
        elif len(va) == 2:
            if triple[1].__contains__('TIME'):
                t = triple[2]
                spv = triple[2]
            else:
                if len(pv) == 2:
                    if '?x0' in va and '?x1' in va:
                        intersect = True
                if 'outer' not in v_triple:
                    v_triple['outer'] = []
                v_triple['outer'].append(triple)
        else:
            print("out of consideration, triple is :", triple)

    if len(pv) > 1 and operator != 'Count':
        if len(pv) == 2:
            print("more than one projected variable")
            if intersect:
                return vtriple2alist(v_triple, t, maxmin, compare, special_va, spv, operator)
            else:
                x0_v_triple = split_v_triple('?x0', v_triple)
                x1_v_triple = split_v_triple('?x1', v_triple)
                return [vtriple2alist(x0_v_triple, t, maxmin, compare, special_va, '?x0', operator), vtriple2alist(x1_v_triple, t, maxmin, compare, special_va, '?x1', operator)]

        else:
            print("out of consideration, projected va is ", pv)

    else:
        return vtriple2alist(v_triple, t, maxmin, compare, special_va, spv, operator)

def split_v_triple(x, v_triple):
    x_v_triple = {}
    aux_list = []
    if x in v_triple:
        x_v_triple[x] = v_triple[x]
    if 'outer' in v_triple:
        for triple in v_triple['outer']:
            if triple[0] == x:
                if triple[2] not in aux_list:
                    aux_list.append(triple[2])
                if 'outer' in x_v_triple:
                    print("something went wrong in split v_triple")
                    return None
                else:
                    x_v_triple['outer'] = [triple]
            elif triple[2] == x:
                if triple[0] not in aux_list:
                    aux_list.append(triple[0])
                if 'outer' in x_v_triple:
                    print("something went wrong in split v_triple")
                    return None
                else:
                    x_v_triple['outer'] = [triple]
    for aux in aux_list:
        if aux in v_triple:
            x_v_triple[aux] = v_triple[aux]
    return x_v_triple
def vtriple2alist(v_triple, t, maxmin, compare, special_va, spv, operator):
    print(v_triple)
    if len(v_triple) == 1 and 'outer' not in v_triple:
        va = list(v_triple.keys())[0]
        if va == 'IS':
            alist = conjunct(v_triple[va])
        else:
            v_nest = nested_alist(v_triple)
            print("v_nest is {} and va is {}".format(v_nest, va))
            alist = v_nest[va]
            alist['h'] = operator
            if t != None:
                alist['t'] = t
            if maxmin != None:
                alist['h'] = maxmin
                alist['v'] = special_va
            if compare != None:
                v = alist['v']
                alist[v] = compare
        return alist
        print("psudo_alist is ", psudo_alist)
    elif len(v_triple['outer']) == 1:
        print("v_triple is ", v_triple)
        alist = simple_alist(operator, spv, v_triple.pop('outer')[0], t)
        if len(v_triple) >= 1:
            v_nest = nested_alist(v_triple)
            for v, nest_list in v_nest.items():
                alist[v] = nest_list
        if maxmin != None:
            alist['h'] = maxmin
            alist['v'] = special_va
        return alist
    else:
        print("length of outer triple is greater than 1, the outer triples are: ", v_triple['outer'])
        return {}

def simple_alist(h ,v, triple, t):
    alist = {'h' : h, 'v' : v, 's': triple[0], 'p' : triple [1], 'o' : triple[2]}
    if t is not None:
        alist['t'] = t
    return alist

def nested_alist(v_triple):
    v_nested = {}
    for va, triples in v_triple.items():
        if len(triples) == 1:
            v_nested[va] = simple_alist('Value', va, triples[0], None)
        else:
            obj = []
            sbj = []
            for triple in triples:
                if triple[0] == va:
                    sbj.append(triple)
                elif triple[2] == va:
                    obj.append(triple)
                else:
                    print("something went wrong when creating nested alist, the triple is: ", triple)
            re_ranked = []
            for triple in obj:
                re_ranked.append(triple)
            for triple in sbj:
                re_ranked.append(triple)
            v_nested[va] = nest(re_ranked, va, 0)
    return v_nested

def nest(re_ranked, va, i):
    alist = simple_alist('Value', va, re_ranked[i], None)
    i += 1
    if i == len(re_ranked):
        return alist
    else:
        alist[va] = nest(re_ranked, va, i)
        return alist
def conjunct(triple_list):
    alist = []
    for i in range(len(triple_list)):
        va = '?x' + str(i)
        obj = triple_list[i][2]
        ntriple = (triple_list[i][0], triple_list[i][1], va)
        sub_alist = simple_alist('IS', va, ntriple, None)
        sub_alist[va] = obj
        alist.append(sub_alist)
    return alist


f = open(test_path, 'a')
ID = 14308
with open('./data/train.json') as fjson:
    client = Client()
    data = json.load(fjson)
    predicate_dict = load_predicate(prediatce_path)
    print(len(data))
    for i in range(ID, len(data)):
        print("The {}th data".format(i))
        SPARQL = data[i]['sparql_wikidata']
        qqnt = data[i]['NNQT_question']
        question = data[i]['question']
        template = data[i]['template']
        # SPARQL = "SELECT DISTINCT ?sbj ?sbj_label WHERE { ?sbj wdt:P31 wd:Q1006311 . ?sbj rdfs:label ?sbj_label . FILTER(CONTAINS(lcase(?sbj_label), 'war')) . FILTER (lang(?sbj_label) = 'en') } LIMIT 25 "
        # qqnt = "Give me {war of national liberation} that contains the word {war} in their name"
        # question = "What are the war of national liberation which start with the letter war"
        # SPARQL = "select ?ent where { ?ent wdt:P31 wd:Q28640 . ?ent wdt:P3618 ?obj } ORDER BY DESC(?obj)LIMIT 5  "
        # qqnt = "What is the {profession} with the {MAX(base salary)}"
        # question = "What profession has the highest base salary"
        sparql = split_sparql(SPARQL)
        pattern = '(?<!\w)\?\w+'
        pv = re.findall(pattern, sparql[0])
        print(sparql)
        #print(pv)
        print('questions is: {}'.format(question))
        print('NNQT questions is {}'.format(qqnt))
        print('template is ', template)
        if question!= None and template != []:
            psudo_alist, nnqt = parse_where(sparql[1], pv, question, qqnt, predicate_dict, client)
            if ('bad', 'data') not in psudo_alist and template != []:
                operator = check_Operator(sparql[0])
                alist = trans2alist(operator, psudo_alist, pv)
                print('alist is ', alist)
                if nnqt:
                    q = qqnt
                else:
                    q = question
                string = "\n Q: " + q + "\n SPARQL: " + SPARQL + "\n ALIST: " + str(alist) + "\n TEMPLATE: " + str(
                    template).strip() + "\n"
                f.writelines(string)
            else:
                print('wrong!!!!!!!!!!!!!!!!')

            print('**********************************************************\n\n')
            # print('operator is {}'.format(operator))



