from django.core.management.base import BaseCommand
from djangorag.lightrag_app.models import ChunkConfig
import uuid


class Command(BaseCommand):
    help = "Create default chunk configuration"

    def handle(self, *args, **options):
        # Check if default config already exists
        existing_default = ChunkConfig.objects.filter(is_default=True).first()
        if existing_default:
            self.stdout.write(self.style.WARNING("Default chunk config already exists"))
            return

        # Create default semantic chunker config
        default_config = ChunkConfig.objects.create(
            id=str(uuid.uuid4()),
            description="Default configuration using Chonkie's SemanticChunker",
            strategy="semantic",
            config={
                "chunk_size": 400,
                "min_chunk_size": 50,
                "max_chunk_size": 1000,
                "overlap": 50,
                "threshold": 0.5,
                "embedding_model": "all-MiniLM-L6-v2",
            },
            is_default=True,
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully created default chunk config: {default_config.name}"
            )
        )
