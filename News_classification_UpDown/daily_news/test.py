from gnews import GNews
import time

def safe_get_top_news(retries=3, wait=2):
    for i in range(retries):
        print(i)
        try:
            google_news = GNews()
            articles = google_news.get_top_news()
            if articles:
                print(articles)
                return articles
        except Exception as e:
            print(f"第 {i+1} 次失敗：{e}")
        time.sleep(wait)
    return []

safe_get_top_news()