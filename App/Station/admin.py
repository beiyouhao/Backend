from django.contrib import admin
from .models import Station


class StationAdmin(admin.ModelAdmin):
    search_fields = ['station_name']

# Register your models here.
admin.site.register(Station, StationAdmin)

