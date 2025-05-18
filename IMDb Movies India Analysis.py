import pandas as pd

# Load the Excel file
#df = pd.read_csv(r'C:\Users\HP\PycharmProjects\pythonProject4\IMDb Movies India.csv')
df = pd.read_csv('IMDb Movies India.csv', encoding='ISO-8859-1')
# or
#df = pd.read_csv('your_file.csv', encoding='latin1')
# Display the first few rows
print(df.head())
print(df.columns)
print(df.shape)
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)
from sklearn.preprocessing import MultiLabelBinarizer
df['Genre_list'] = df['Genre'].str.split(',')
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['Genre_list'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
df = pd.concat([df, genre_df], axis=1)
cat_cols = ['Director', 'Genre','Actor 1', 'Actor 2', 'Actor 3']
for col in cat_cols:
    freq = df[col].value_counts()
    df[col+ '_FE']= df[col].map(freq)
print(df[[col + '_FE' for col in cat_cols]].head())
df.to_csv("movie_features_final.csv", index = False)




