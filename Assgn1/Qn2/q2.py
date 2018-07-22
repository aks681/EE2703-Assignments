import math

array = []
result = []
array.append(0.2)
result.append(0.2)

for i in range(1,1000):
    array.append(((array[i-1] + math.pi)*100) - (int)((array[i-1] + math.pi)*100))
    result.append('%.4f' %array[i])

for i in range(0,len(result)):
    print '%d: %s' %(i+1, result[i])
