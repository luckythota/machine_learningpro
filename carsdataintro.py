#cars dataset
a=open(r"E:\cpp programs\New folder\New folder\cars1.csv","r")
f=a.readlines()
#print(f)
l=[]
for i in f:
    l.append(i.split(','))
#print(l)
d={}
##for i in l:
##    if i[-1] not in d.keys():
##        d[i[-1]]=1
##    else:
##        d[i[-1]]+=1
##print(d)
c=0
##for i in l:
##    if i[8][:-1]=='Europe' and i[7]>'70':
##        c+=1
##print(c)
#to find the spaces which row and column
##for i in range(len(l)):
##    for j in range(len(l[i])):
##        if l[i][j]=='':
##            print('row is:{},column is:{}'.format(i,j))
##                  
#print(c)
#to find the no. of spaces count for each column
##for i in range(len(l[0])):
##    d[i]=0
##for i in l:
##    for j in range(len(i)):
##        if i[j]=='':
##            d[j]+=1
##print(d)
##c=0
##for i in l:
##    if i[-1] not in d.keys():
##        d[i[0]]={}
##        if i[-1] not in d[i[0]].keys():
##            d[i[0]][i[-1]]=1
##        else:
##            d[i[0]][i[-1]]+=1
##            c+=1
##    else:
##        if i[-1] not in d[i[0]].keys():
##            d[i[0]][i[-1]]=1
##        else:
##            d[i[0]][i[-1]]+=1
##v=[]
##n=[]
##v.append(d.keys())
##n.append(d.values())
##print(v)
##print(n)
##import matplotlib.pyplot as plt
##plt.plot(v,n)
##plt.show()
##            
        
for i in l:
    if i[-1] not in d.keys():
        d[i[-1]]=1
    else:
        d[i[-1]]+=1
print(d)
import matplotlib.pyplot as plt
#v=d.keys()
labels=list(d.keys())
n=list(d.values())
h=[2,3]
plt.bar(h,n,tick_label=labels)
plt.show()

##cdic={}
##for i in l:
##    a=i[-1][:-1]
##    b=i[0]
##    if a not in cdic:
##        cdic[a]=[b]
##    else:
##        cdic[a].append(b)
##print(cdic)
            
    
        
        
    
