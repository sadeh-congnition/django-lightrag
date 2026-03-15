import json
from django.test import TestCase, Client
from django.contrib.auth.models import User


class LightRAGAPITest(TestCase):
    """Test cases for LightRAG API endpoints"""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/api/lightrag/health/")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "lightrag-django")
