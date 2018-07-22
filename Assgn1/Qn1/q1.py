a1=1
a0=1
print '1: %d' %a0
print '2: %d' %a1

for k in range(3,11):
    a2=a0+a1
    a0=a1
    a1=a2
    print '%d: %d' %(k, a1)
