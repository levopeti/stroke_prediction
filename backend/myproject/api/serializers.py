from rest_framework import serializers
from base.models import Item


class ItemSerializer(serializers.ModelSerializer):
    class Meta(object):
        model = Item
        fields = '__all__'

