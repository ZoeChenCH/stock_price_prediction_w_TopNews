from gnews import GNews
from dateutil import parser
import csv
from datetime import datetime, timedelta
import os

def fetch_top_news(days_ago=1, output_dir="daily_news"):
    target_date = datetime.now() - timedelta(days=days_ago)
    date_str = target_date.strftime("%Y-%m-%d")

    google_news = GNews(language='en', country='US', max_results=100)

    os.chdir('/Users/zhao-weichen/Zoe/Strock_tracking/News_classification_UpDown')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"news_{date_str}.csv")

    print(f"開始抓取 {date_str} 的頭版新聞")

    top_articles = google_news.get_top_news()
    news_data = []
    count = 0

    for a in top_articles:
        pub_date = a.get("published date")
        try:
            pub_dt = parser.parse(pub_date)
        except:
            continue

        # 只收當天的新聞
        if pub_dt.date() == target_date.date():
            news_data.append([
                date_str,
                "TOP_NEWS",
                a.get("title", ""),
                a.get("description", "")
            ])
            count += 1
            if count >= 20:
                break

    print(f" 完成！共儲存 {len(news_data)} 則頭版新聞")

    # 儲存成 CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "source", "title", "description"])
        writer.writerows(news_data)


#  執行
if __name__ == "__main__":
    fetch_top_news(days_ago=1)
