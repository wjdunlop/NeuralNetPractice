f = open('roswell.txt', encoding = 'utf-8')
lens = []
for l in f.readlines():
    lens.append(len(l))

from matplotlib import pyplot as ppl
ppl.hist(lens)
ppl.show()