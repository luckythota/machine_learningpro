#import matplotlib.pyplot as plt
#v=[1,4,9,16,25,36]
#n=[1,8,27,64,125,216]
#d=[1,3,5,7,9,11]
#plt.plot(v,n,c='g',linewidth=5,linestyle='dotted',marker='*',mfc='blue',mec='g',label='squares')
#plt.plot(v,d,c='r',linewidth=5,linestyle='dashed',marker='o',mfc='blue',mec='r',label='cubes')
##plt.title('squares vs cubes')
##plt.xlabel('squares')
##plt.ylabel('cubes')
#plt.scatter(v,n,marker='^')
##plt.legend()
##plt.show()
#----------------------------------------
## bar graph
##import matplotlib.pyplot as plt
##v=[3,4,5,2]
##c=[1,2,3,4]
##s=['2020','2021','2022','2023']
##plt.bar(c,v,tick_label=s,color='black')
##plt.barh(c,v,tick_label=s,color='black')
##plt.show()

#----------------------------------------------------------------------
## same space divide into two parts and we can draw graphs individually
##import matplotlib.pyplot as plt
##v=[3,4,5,2]
##c=[2,4,1,5]
##d=[4,1,2,5]
##
##plt.subplot(2,1,1)
##plt.plot(v,c)
##
##plt.subplot(2,1,2)
##plt.plot(v,d,c='g')
##
##plt.show()
#-------------------------------------------------------------------------
#pie chart
import matplotlib.pyplot as plt
a=[2,6,7,9]
b=["c","python","cpp","java"]
c=["red","blue","green","yellow"]
plt.pie(a,labels=b,startangle=90,explode=(0,0.2,0,0),autopct='%1.2f',colors=c)
# explode is used for to highlight the particular portion
plt.legend()
plt.savefig(r'C:\Users\ram naidu\OneDrive\Desktop\ram\graph1.jpg')
plt.show()





