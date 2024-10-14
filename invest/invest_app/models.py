from django.db import models

class TweetSentiment(models.Model):
    tweet_text = models.TextField()
    prediction = models.CharField(max_length=50,  default='neutral')   # 'Tweet Text' can be large, so we use TextField
    sentiment = models.CharField(max_length=50)  # 'Sentiment' can be a short string

    def __str__(self):
        return f"{self.tweet_text[:50]} - {self.sentiment}"  # Shortened display




class Count_samples_for_retrain(models.Model):
    samples_number =   models.IntegerField()