"""EduMIND Label Studio ML Backend Package.

This package exposes the ``EduMINDMLBackend`` class for use by the
``label-studio-ml`` CLI.  The CLI discovers the model class by importing
this ``__init__.py`` and looking for a ``LabelStudioMLBase`` subclass.
"""

from label_studio_backend.model import EduMINDMLBackend  # noqa: F401

__all__ = ["EduMINDMLBackend"]
