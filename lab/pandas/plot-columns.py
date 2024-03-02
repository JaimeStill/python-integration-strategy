import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plot() plots all columns

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))

df = pd.DataFrame(
    np.random.randn(1000, 4), index=ts.index, columns=["A", "B", "C", "D"]
)

df = df.cumsum()
df.plot()
plt.legend(loc="best")
plt.show()