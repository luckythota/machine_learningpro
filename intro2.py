c=open(r"E:/cpp programs/New folder/New folder/diabetes_data_upload.csv","r")
f=c.readlines()
###print(f)
c.close()
l=[]
for i in f:
    l.append(i.split(','))
#d={}
##for calculating how many positive and  negative
##for i in l:
##    if i[-1] not in d.keys():
##             d[i[-1][:-1]]=1  #{'positive':1}
##    else:
##       d[i[-1]]+=1   #{'positive':2}
##for calculating how many male and  female
c=0
d={}
##for i in l:
##     if i[0] not in d.keys() and i[0]<='30':
##         d[i[0]]=1
##         c+=1
##     elif i[0]<='30':
##         d[i[0]]+=1
##         c+=1

##for i in l:
##    if i[0]<='30' and i[-1]=='Positive\n':
##        c+=1

##for i in l:
##    if i[0]>'40' and i[-1]=='Negative\n':
##        c+=1

#print(d)
##mcd=0
##mc=0
##fcd=0
##fc=0
##for i in l:
##    if i[0]=='Male' and i[-1]=='Positive\n':
##        mcd+=1
##    elif i[0]=='Male' and i[-1]=='Negative\n':
##        mc+=1
##    elif i[0]=='Female' and i[-1]=='Positive\n':
##        fcd+=1
##    else:
##        fc+=1
##        
##print(mcd)
##print(mc)
##print(fcd)
##print(fc)
for i in l:
    if i[1] not in d.keys():
             d[i[1]]={}
             if i[-1] not in d[i[1]].keys():
                    d[i[1]][i[-1]]=1  
             else:
                d[i[1]][i[-1]]+=1
    else:
        if i[-1] not in d[i[1]].keys():
            d[i[1]][i[-1]]=1
        else:
            d[i[1]][i[-1]]+=1
##for i in d:
##    print(i,d[i])
#c=0
#d={}
#to find the empty spaces followed by row and column
##for i in range(len(l)):
##    for j in range(len(l[i])):
##        if (l[i][j]== ''):
##            print('row is :{},column is:{}'.format(i,j))
##            
#print(c)
#print(d)
# for each column count the spaces 
##w={}
##for i in range(len(l[0])):
##    w[i]=0
##for i in l:
##    for j in range(len(i)):
##        if i[j]=='':
##            w[j]+=1
##print(w)
               
               
               
             





