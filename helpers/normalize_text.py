import unicodedata

def normalize_text(text):
    """Normaliza texto (remove acentos e cedilhas)."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')