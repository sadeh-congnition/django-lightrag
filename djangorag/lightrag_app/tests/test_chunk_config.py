from django.test import TestCase
from djangorag.lightrag_app.models import ChunkConfig
import uuid


class ChunkConfigTest(TestCase):
    def test_create_chunk_config(self):
        """Test creating a ChunkConfig instance"""
        chunk_config = ChunkConfig.objects.create(
            id=str(uuid.uuid4()),
            description="Test configuration",
            strategy="semantic",
            config={"chunk_size": 500, "threshold": 0.7},
        )

        self.assertEqual(chunk_config.strategy, "semantic")
        self.assertFalse(chunk_config.is_default)
        self.assertEqual(chunk_config.config["chunk_size"], 500)

    def test_semantic_chunker_default_config(self):
        """Test that semantic chunker gets default configuration"""
        chunk_config = ChunkConfig.objects.create(
            id=str(uuid.uuid4()), description="Semantic Test", strategy="semantic"
        )

        # Check that default config is applied
        self.assertIn("chunk_size", chunk_config.config)
        self.assertIn("threshold", chunk_config.config)
        self.assertIn("embedding_model", chunk_config.config)
        self.assertEqual(chunk_config.config["chunk_size"], 400)

    def test_chunk_config_str_representation(self):
        """Test string representation of ChunkConfig"""
        chunk_config = ChunkConfig.objects.create(
            id=str(uuid.uuid4()), description="Test Config", strategy="semantic"
        )

        expected_str = "(semantic)"
        self.assertEqual(str(chunk_config), expected_str)

    def test_unique_name_constraint(self):
        """Test that multiple configs can have same description (no unique constraint)"""
        config_id = str(uuid.uuid4())
        ChunkConfig.objects.create(
            id=config_id, description="Unique Name", strategy="semantic"
        )

        # Should succeed since there's no unique constraint on description
        second_config = ChunkConfig.objects.create(
            id=str(uuid.uuid4()), description="Unique Name", strategy="recursive"
        )

        # Verify both configs exist
        self.assertEqual(
            ChunkConfig.objects.filter(description="Unique Name").count(), 2
        )
        self.assertEqual(second_config.strategy, "recursive")
