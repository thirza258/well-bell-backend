from django.db import models

# Create your models here.
class FormData(models.Model):
    age = models.CharField(max_length=10)
    gender = models.CharField(max_length=10)
    mobileOS = models.CharField(max_length=20)
    educationUse = models.CharField(max_length=100)
    activities = models.JSONField()
    helpfulForStudying = models.CharField(max_length=200)
    educationalApps = models.JSONField()
    dailyUsage = models.CharField(max_length=20)
    performanceImpact = models.CharField(max_length=200, null=True, blank=True)
    usageDistraction = models.CharField(max_length=200)
    usefulFeatures = models.JSONField()
    beneficialSubjects = models.JSONField()
    usageSymptoms = models.JSONField()
    symptomFrequency = models.CharField(max_length=100)
    healthPrecautions = models.JSONField()

    def __str__(self):
        return f"FormData: {self.age}, {self.gender}"