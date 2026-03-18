from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("lightrag_app", "0006_remove_file_fields"),
    ]

    operations = [
        migrations.DeleteModel(
            name="DocumentStatus",
        ),
    ]
