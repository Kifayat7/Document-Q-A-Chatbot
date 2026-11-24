"""Custom exception types so the UI can show friendly, specific messages."""


class DocuMindError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigError(DocuMindError):
    """Missing or invalid configuration (e.g. no API key)."""


class ValidationError(DocuMindError):
    """User input (file, question) failed validation."""


class DuplicateDocumentError(DocuMindError):
    """The same PDF content was already uploaded by this user."""


class DocumentLoadError(DocuMindError):
    """A PDF could not be read or contained no extractable text."""
