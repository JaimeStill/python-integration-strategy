import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# We use the standard convention for referencing the matplotlib API:
# import matplotlib.pyplot as plt
# plt.close("all")

# The plt.close method is used to close a figure window:
ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()