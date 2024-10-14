from django.contrib import admin

# Register your models here.
from .models import TweetSentiment ,   Count_samples_for_retrain  # Import your model

# Register the model with the admin site
admin.site.register(TweetSentiment)
admin.site.register(Count_samples_for_retrain)