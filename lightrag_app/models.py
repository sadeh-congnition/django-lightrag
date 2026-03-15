import json
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.exceptions import ValidationError


class Workspace(models.Model):
    """Workspace for data isolation and multi-tenancy"""
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='workspaces')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'lightrag_workspaces'
        ordering = ['name']

    def __str__(self):
        return self.name


class Document(models.Model):
    """Document representation in the RAG system"""
    id = models.CharField(max_length=255, primary_key=True)  # MD5 hash or UUID
    title = models.CharField(max_length=500, blank=True)
    content = models.TextField()
    file_path = models.CharField(max_length=1000, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    track_id = models.CharField(max_length=100, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lightrag_documents'
        indexes = [
            models.Index(fields=['track_id']),
        ]

    def __str__(self):
        return f"{self.title or self.id[:50]}..."


class DocumentStatus(models.Model):
    """Track document processing status"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('processed', 'Processed'),
        ('failed', 'Failed'),
    ]

    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='status')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    chunks_count = models.IntegerField(default=0)
    chunks_list = models.JSONField(default=list, blank=True)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lightrag_document_status'
        indexes = [
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"{self.document.id} - {self.status}"


class TextChunk(models.Model):
    """Text chunks from document processing"""
    id = models.CharField(max_length=255, primary_key=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField()
    tokens = models.IntegerField()
    chunk_order_index = models.IntegerField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lightrag_text_chunks'
        ordering = ['document', 'chunk_order_index']
        indexes = [
            models.Index(fields=['document']),
        ]

    def __str__(self):
        return f"Chunk {self.chunk_order_index} of {self.document.id}"


class Entity(models.Model):
    """Knowledge graph entities"""
    id = models.CharField(max_length=255, primary_key=True)
    name = models.CharField(max_length=500, db_index=True)
    entity_type = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    source_ids = models.JSONField(default=list, blank=True)  # List of chunk IDs
    file_paths = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lightrag_entities'
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['entity_type']),
        ]

    def __str__(self):
        return f"{self.name} ({self.entity_type})"


class Relation(models.Model):
    """Knowledge graph relationships"""
    id = models.CharField(max_length=255, primary_key=True)
    source_entity = models.ForeignKey(Entity, on_delete=models.CASCADE, related_name='outgoing_relations')
    target_entity = models.ForeignKey(Entity, on_delete=models.CASCADE, related_name='incoming_relations')
    relation_type = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    source_ids = models.JSONField(default=list, blank=True)  # List of chunk IDs
    file_paths = models.JSONField(default=list, blank=True)
    weight = models.FloatField(default=1.0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lightrag_relations'
        indexes = [
            models.Index(fields=['source_entity', 'target_entity']),
            models.Index(fields=['relation_type']),
        ]

    def __str__(self):
        return f"{self.source_entity.name} -> {self.relation_type} -> {self.target_entity.name}"


class EntityChunk(models.Model):
    """Mapping between entities and chunks"""
    id = models.CharField(max_length=255, primary_key=True)
    entity = models.ForeignKey(Entity, on_delete=models.CASCADE, related_name='entity_chunks')
    chunk = models.ForeignKey(TextChunk, on_delete=models.CASCADE, related_name='entity_chunks')
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'lightrag_entity_chunks'
        indexes = [
            models.Index(fields=['entity']),
            models.Index(fields=['chunk']),
        ]

    def __str__(self):
        return f"Entity {self.entity.name} in Chunk {self.chunk.id}"


class RelationChunk(models.Model):
    """Mapping between relations and chunks"""
    id = models.CharField(max_length=255, primary_key=True)
    relation = models.ForeignKey(Relation, on_delete=models.CASCADE, related_name='relation_chunks')
    chunk = models.ForeignKey(TextChunk, on_delete=models.CASCADE, related_name='relation_chunks')
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'lightrag_relation_chunks'
        indexes = [
            models.Index(fields=['relation']),
            models.Index(fields=['chunk']),
        ]

    def __str__(self):
        return f"Relation {self.relation.id} in Chunk {self.chunk.id}"


class VectorEmbedding(models.Model):
    """Vector embeddings for entities, relations, and chunks"""
    VECTOR_TYPES = [
        ('entity', 'Entity'),
        ('relation', 'Relation'),
        ('chunk', 'Chunk'),
    ]

    id = models.CharField(max_length=255, primary_key=True)
    vector_type = models.CharField(max_length=20, choices=VECTOR_TYPES)
    content_id = models.CharField(max_length=255)  # Reference to entity/relation/chunk ID
    embedding = models.JSONField()  # Store as JSON array
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lightrag_vector_embeddings'
        indexes = [
            models.Index(fields=['vector_type', 'content_id']),
        ]

    def __str__(self):
        return f"{self.vector_type} embedding for {self.content_id}"


class CacheEntry(models.Model):
    """Cache for LLM responses and other expensive operations"""
    CACHE_TYPES = [
        ('llm_response', 'LLM Response'),
        ('embedding', 'Embedding'),
        ('entity_extraction', 'Entity Extraction'),
        ('relation_extraction', 'Relation Extraction'),
    ]

    id = models.CharField(max_length=255, primary_key=True)
    cache_type = models.CharField(max_length=50, choices=CACHE_TYPES)
    cache_key = models.CharField(max_length=1000)
    cache_value = models.JSONField()
    expires_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lightrag_cache'
        unique_together = [['cache_key']]
        indexes = [
            models.Index(fields=['cache_type']),
            models.Index(fields=['expires_at']),
        ]

    def __str__(self):
        return f"{self.cache_type} cache: {self.cache_key[:50]}..."

    def is_expired(self):
        """Check if cache entry is expired"""
        if self.expires_at is None:
            return False
        return timezone.now() > self.expires_at


class ProcessingJob(models.Model):
    """Track background processing jobs"""
    JOB_TYPES = [
        ('document_ingestion', 'Document Ingestion'),
        ('entity_extraction', 'Entity Extraction'),
        ('relation_extraction', 'Relation Extraction'),
        ('embedding_generation', 'Embedding Generation'),
    ]

    JOB_STATUSES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]

    id = models.CharField(max_length=255, primary_key=True)
    job_type = models.CharField(max_length=50, choices=JOB_TYPES)
    status = models.CharField(max_length=20, choices=JOB_STATUSES, default='pending')
    input_data = models.JSONField(default=dict)
    output_data = models.JSONField(default=dict)
    error_message = models.TextField(blank=True)
    progress = models.FloatField(default=0.0)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lightrag_processing_jobs'
        indexes = [
            models.Index(fields=['job_type']),
            models.Index(fields=['status']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.job_type} job {self.id} - {self.status}"
