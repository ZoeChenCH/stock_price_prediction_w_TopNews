import os
import pandas as pd
os.environ["TRANSFORMERS_NO_TF"] = "1"

def load_daily_news(folder_path):
    all_news = []

    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.csv')],
        key=lambda x: pd.to_datetime(x.replace('news_', '').replace('.csv', ''))
    )

    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)

        try:
            df = pd.read_csv(file_path)
            date_str = file_name.replace('news_', '').replace('.csv', '')
            date = pd.to_datetime(date_str).date()

            df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
            df['date'] = date

            all_news.append(df[['date', 'text']])
        except Exception as e:
            print(f"can not read {file_name}: {e}")


    df_all_news = pd.concat(all_news, ignore_index=True)
    daily_news = df_all_news.groupby('date')['text'].apply(lambda x: ' '.join(x)).reset_index()

    return daily_news

folder_path = './daily_news'
daily_news_df = load_daily_news(folder_path)

import yfinance as yf
import pandas as pd

start_date = daily_news_df['date'].min().strftime('%Y-%m-%d')
end_date = daily_news_df['date'].max().strftime('%Y-%m-%d')

twii = yf.download('^TWII', start=start_date, end=end_date)
twii['Change'] = twii['Close'].diff()/twii['Close'].shift(1)*100

def classify_movement(change):
    if change > 0.1:
        return 'bullish'
    elif change < -0.1:
        return 'bearish'
    else:
        return 'flat'

twii['Movement'] = twii['Change'].apply(classify_movement)
twii = twii.reset_index()
twii.columns = twii.columns.get_level_values(0)
twii = twii[['Date', 'Close', 'Change', 'Movement']]
twii = twii.rename(columns={'Date': 'date'})
daily_news_df = daily_news_df.copy()
twii['date'] = pd.to_datetime(twii['date']).dt.date
daily_news_df['date'] = pd.to_datetime(daily_news_df['date']).dt.date

merged_df = pd.merge(twii, daily_news_df, on='date', how='left')
merged_df['news_prev2'] = merged_df['text'].shift(2)

train_df = merged_df.dropna(subset=['news_prev2', 'Movement'])[['news_prev2', 'Movement']]
train_df = train_df.rename(columns={'news_prev2': 'text', 'Movement': 'label'})

print(twii.columns)
print(daily_news_df.columns)
print(twii.dtypes)
print(daily_news_df.dtypes)

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
train_df['label_id'] = le.fit_transform(train_df['label'])  # eg. bullish → 0, flat → 1, bearish → 2

texts = train_df['text'].tolist()
labels = train_df['label_id'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = NewsDataset(X_train, y_train, tokenizer)
val_dataset = NewsDataset(X_val, y_val, tokenizer)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

from sklearn.metrics import accuracy_score, classification_report

predictions = trainer.predict(val_dataset)
pred_labels = predictions.predictions.argmax(axis=-1)
true_labels = predictions.label_ids

acc = accuracy_score(true_labels, pred_labels)
print(f"accuracy：{acc:.4f}")

print("classification_report：")
print(classification_report(true_labels, pred_labels, target_names=le.classes_))
