import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import chart_studio
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import chart_studio.plotly as py
import cufflinks as cf
cf.go_offline()
df=pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
#print(df.head())
print(df.iplot())



