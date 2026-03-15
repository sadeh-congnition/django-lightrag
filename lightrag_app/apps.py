from django.apps import AppConfig


class LightragAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'djangorag.lightrag_app'
    verbose_name = 'LightRAG'

    def ready(self):
        from . import signals
