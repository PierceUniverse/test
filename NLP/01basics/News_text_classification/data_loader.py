import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


train_df = pd.read_csv('../../../../data/train_set.csv', sep='\t', nrows=100)
# print(train_df.head())

train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
# print(train_df['text_len'].describe())

_ = plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")
# plt.show()

train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
# plt.show()


all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
# print(len(word_count))
# print(word_count[0])
# print(word_count[-1])



train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d: int(d[1]), reverse=True)
print(word_count[0])
print(word_count[1])
print(word_count[2])