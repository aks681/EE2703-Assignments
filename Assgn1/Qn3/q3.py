import string

with open('the-hound-of-the-baskervilles.txt',"r") as f:
    contents=f.read()

words = contents.split()
f = {}

# The following loop takes the words from the text file, removes the punctuation from the end,
# makes it lower case and creates a dictionary with the number of occurrences of each such word.

for i in range(0,len(words)):
    words[i] = words[i].strip(string.punctuation).lower()
    if words[i] in f:
        f[words[i]]+=1
    else:
        f[words[i]]=1

# The following loop prints the 10 most occuring words (all without punctuations and converted to lower case)
i=0
for k in sorted(f,key=f.get,reverse=True):
    print k,f[k]
    i = i+1
    if (i==10):
        break
