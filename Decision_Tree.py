import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
df =pandas.read_csv("D:\Machine Learning\shows.csv")
d={'UK':0,'USA':1,'N':2}
df['Nationality']=df['Nationality'].map(d)
d={'YES':1,'NO':0}
df['Go']=df['Go'].map(d)

features=['Age','Experience','Rank','Nationality']
x=df[features]
y=df['Go']

dtree=DecisionTreeClassifier()
dtree=dtree.fit(x,y)
data=tree.export_graphviz(dtree,out_file=None,feature_names=features)
graph=pydotplus.graph_from_dot_data(data)
graph.write_png('D:\Machine Learning\mydecisiontree.png')
img=pltimg.imread('D:\Machine Learning\mydecisiontree.png')
imgplot=plt.imshow(img)
plt.show()