import seaborn as sns
df = sns.load_dataset("penguins")
# df["species"] = df["species"].map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})
df["target"] = df["species"].map({"Adelie": 1, "Chinstrap": 0, "Gentoo": 0})
df.drop("species", axis=1, inplace=True)

print(df.head())
print(df["target"].value_counts())
df.to_csv("./data/penguins.csv", index=False)