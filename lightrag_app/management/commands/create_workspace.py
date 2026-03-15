"""
Django management command to create a LightRAG workspace.
"""

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from lightrag_app.models import Workspace


class Command(BaseCommand):
    help = 'Create a LightRAG workspace'

    def add_arguments(self, parser):
        parser.add_argument(
            'name',
            type=str,
            help='Workspace name'
        )
        parser.add_argument(
            '--description',
            type=str,
            default='',
            help='Workspace description'
        )
        parser.add_argument(
            '--user',
            type=str,
            default='admin',
            help='Username for workspace owner (default: admin)'
        )
        parser.add_argument(
            '--inactive',
            action='store_true',
            help='Create workspace as inactive'
        )

    def handle(self, *args, **options):
        name = options['name']
        description = options.get('description', '')
        username = options.get('user', 'admin')
        is_active = not options.get('inactive', False)

        # Get user
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise CommandError(f'User "{username}" does not exist')

        # Check if workspace already exists
        if Workspace.objects.filter(name=name).exists():
            raise CommandError(f'Workspace "{name}" already exists')

        # Create workspace
        try:
            workspace = Workspace.objects.create(
                name=name,
                description=description,
                created_by=user,
                is_active=is_active
            )

            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully created workspace "{name}" (ID: {workspace.id})'
                )
            )

            if not is_active:
                self.stdout.write(
                    self.style.WARNING('Workspace created as inactive')
                )

        except Exception as e:
            raise CommandError(f'Failed to create workspace: {e}')
