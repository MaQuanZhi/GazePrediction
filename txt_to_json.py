import json

dict_result = {}
dict_line = {}
with open('result_test.txt','r') as f:
    data = f.read().split('\n')
    for line in data[:-1]:
        l = line.split(',')
        print(l)
        dict_line.update({l[1]:list(map(float,l[2:]))})
        dict_result.update({l[0]:dict_line})

json.dump(dict_result,open('result.json','w'))

# print(dict_result)
