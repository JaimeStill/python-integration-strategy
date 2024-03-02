import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Series - a one-dimensional labeled array holding data of any type
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# DataFrame - a two-dimensional data structure that holds data like a
# two-dimension array or a table with rows and columns
dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print (df)

# create a DataFrame by passing a dictionary of objects where the keys
# are the column labels and the values are the column values
df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo"
    }
)
print(df2)
print(df2.dtypes)

# Use DataFrame.head() and DataFrame.tail() to view the top and bottom
# rows of the frame respectively:
print(df.head())
print(df.tail(3))

# Display the DataFrame.index or DataFrame.columns
print(df.index)
print(df.columns)

# Return a NumPy representation of the underlying data with
# DataFrame.to_numpy() without the index or column labels:
print(df.to_numpy())

# describe() shows a quick statistic summary of your data:
print(df.describe())

# Transposing your data:
print(df.T)

# DataFrame.sort_index() sorts by an axis:
print(df.sort_index(axis=1, ascending=False))

# DataFrame.sort_values() sorts by values:
print(df.sort_values(by="B"))

# For a DataFrame, passing a single label selects a column
# and yields a Series equivalent to df.A:
print(df["A"])

# For a DataFrame, passing a slice : selects matching rows:
print(df[0:3])
print(df["20130102":"20130104"])

# Selecting a row matching a label:
print(df.loc[dates[0]])

# Selecting all rows (:) with a select column labels:
print(df.loc[:, ["A", "B"]])

# For label slicing, both endpoints are included:
print(df.loc["20130102": "20130104", ["A", "B"]])

# Selecting a single row and column label return a scalar:
print(df.loc[dates[0], "A"]) # type: ignore

# For getting fast access to a scalar (equivalent to above):
print(df.at[dates[0], "A"])

# Select via the position of the passed integers:
print(df.iloc[3])

# Integer slices act similar to NumPy/Python:
print(df.iloc[3:5, 0:2])

# Lists of integer position locations
print(df.iloc[[1, 2, 4], [0, 2]])

# For slicing rows explicitly
print(df.iloc[1:3, :])

# For slicing columns explicitly
print(df.iloc[:, 1:3])

# For getting a value explicitly
print(df.iloc[1, 1])

# For fast access to a scalar (equivalent to above):
print(df.iat[1, 1])

# Select rows where df.A is greater than 0
print(df[df["A"] > 0])

# Selecting values from a DataFrame here a boolean condition is met:
print(df[df > 0])

# Using isin() method for filtering
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
print (df2[df2["E"].isin(["two", "four"])])

# Setting a new column automatically aligns the data by the indexes:
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
df["F"] = s1

# Setting values by label
df.at[dates[0], "A"] = 0

# Setting values by position:
df.iat[0, 1] = 0

# Setting by assigning with a NumPy array:
df.loc[:, "D"] = np.array([5] * len(df))

# Result of prior setting operations:
print(df)

# A where operation with setting:
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

# Re-indexing allows you to change/add/delete the index on a specified axis.
# This returns a copy of the data:
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0] : dates[1], "E"] = 1
print(df1)

# DataFrame.dropna() drops any rows that have missing data:
print(df1.dropna(how="any"))

# DataFrame.fillna() fills missing data
print(df1.fillna(value=5))

# isna() gets the boolean mask where values are nan
print(pd.isna(df1))

# Calculate the mean value for each column:
print(df.mean())

# Calculate the mean value for each row:
print(df.mean(axis=1))

# Operating with another Series or DataFrame with a different index
# or column will align the result with the union of the index or
# column labels. In addition, pandas automatically broadcasts along
# the specified dimension and will fill unaligned labels with np.nan:
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print(s)
print(df.sub(s, axis="index"))

# DataFrame.agg() and DataFrame.transform() applies a user defined function that
# reduces or broadcasts its result respectively:
print(df.agg(lambda x: np.mean(x) * 5.6))
print(df.transform(lambda x: x * 101.2))

# Value counts:
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())

# Series is equipped with a set of string processing methods in the str
# attribute that makes it easy to operate on each element of the array,
# as in the code snippet below:
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
print(s.str.lower())

# pandas provides various facilities for easily combining together
# Series and DataFrame objects with various kinds of set logic for
# the indexes and relational algebra functionality in case of join
# / merge-type operations.

# Concatenating pandas objects together row-wise with concat():
df = pd.DataFrame(np.random.randn(10, 4))
print(df)

pieces = [df[:3], df[3:7], df[7:]]
print(pd.concat(pieces))

# merge() enables SQL style join types along specific columns:
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
print(left)
print(right)
print(pd.merge(left, right, on="key"))

# merge() on unique keys
left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})
print(left)
print(right)
print(pd.merge(left, right, on="key"))

# By "group by" we are referring to a process involving one
# or more of the following steps:
# Splitting the data into groups based on some criteria
# Applying a function to each group independently
# Combining the results into a data structure

df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)

print(df)

# Grouping by a column label, selecting column labels, and then
# applying the DataFrameGroupBy.sum() function to the resulting group:
print(df.groupby("A")[["C", "D"]].sum())

# Grouping by multiple columns label forms MultiIndex:
print(df.groupby(["A", "B"]).sum())

# Stack

arrays = [
   ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
   ["one", "two", "one", "two", "one", "two", "one", "two"],
]

index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df2 = df[:4]
print(df2)

# The stack() method "compresses" a level in the DataFrame's columns:
stacked = df2.stack(future_stack=True)
print(stacked)

# With a "stacked" DataFrame or Series (having a MultiIndex as index),
# the inverse operation of stack() is unstack(), which by default
# unstacks the last level:
print(stacked.unstack())
print(stacked.unstack(1))
print(stacked.unstack(0))

# Pivot tables:
df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)
print(df)

# pivot_table() pivots a DataFrame specifying the values, index, and columns:
print(pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"]))

# pandas has simple, powerful, and efficient functionality for performing
# resampling operations during frequency conversion (e.g., converting
# secondly data into 5-minutely data). This is extremely common in, but
# not limited to, financial applications.

rng = pd.date_range("1/1/2012", periods=100, freq="s")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts.resample("5Min").sum())

# Series.tz_localize() localizes a time series to a time zone:
rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)

ts_utc = ts.tz_localize("UTC")
print(ts_utc)

# Series.tz_convert() converts a timezones aware time series to another time zone:
print(ts_utc.tz_convert("US/Eastern"))

# Adding a non-fixed duration (BusinessDay) to a time series:
print(rng)
print(rng + pd.offsets.BusinessDay(5))

# pandas can include categorical data in a DataFrame:
df = pd.DataFrame(
    {
        "id": [
            1, 2, 3, 4, 5, 6
        ],
        "raw_grade": [
            "a", "b", "b", "a", "a", "e"
        ]
    }
)

# Converting the raw grades to a categorical data type
df["grade"] = df["raw_grade"].astype("category")
print(df["grade"])

# Rename the categories to more meaningful names:
new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)

# Reorder the categories and simultaneously add the missing categories
# (methods under Series.cat() return a new Series by default):
df["grade"] = df["grade"].cat.set_categories(
    [ "very bad", "bad", "medium", "good", "very good" ]
)
print(df["grade"])

# Sorting is per order in the categories, not lexical order:
print(df.sort_values(by="grade"))

# Grouping by a categorical column with observed=False also shows empty categories:
print(df.groupby("grade", observed=False).size())

# Create data directory:
datapath = (Path.cwd() / 'data')

if not datapath.exists():
    datapath.mkdir(parents=True)

# Initialize file paths
csvpath = (datapath / "foo.csv")
parquetpath = (datapath / "foo.parquet")
xlsxpath = (datapath / "foo.xlsx")

if csvpath.exists():
    Path.unlink(csvpath)

if parquetpath.exists():
    Path.unlink(parquetpath)

if xlsxpath.exists():
    Path.unlink(xlsxpath)

# Writing to a CSV file using DataFrame.to_csv()
df = pd.DataFrame(np.random.randint(0, 5, (10, 5)))
df.to_csv(csvpath)

# Reading from a csv file using read_csv()
print(pd.read_csv(csvpath))

# Writing to a Parquet file:
df.to_parquet(parquetpath)

# Reading from a Parquet file Store using read_parquet():
print(pd.read_parquet(parquetpath))

# Writing to an excel file using DataFrame.to_excel():
df.to_excel(xlsxpath, sheet_name="Foo Data")

# Reading from an excel file using read_excel():
print(pd.read_excel(xlsxpath, "Foo Data", index_col=None, na_values=["NA"]))