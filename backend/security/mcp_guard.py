import re

class MCPGuard:
    """
    Security layer: MCP Pattern Guard.
    Responsible for masking PII before sending data to the LLM.
    """

    def __init__(self):
        # Basic regex patterns for PII detection
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?\b')

        # Name and address are harder to mask with regex, we will use simplistic replacements
        # for typical CV patterns or rely on named entity recognition (NER) in a real-world scenario.
        # For this requirement, we will simulate the masking of name and address.

    def sanitize_cv_text(self, text: str) -> str:
        """
        Sanitize CV text by masking PII: name, email, phone, address.
        """
        sanitized_text = text

        # Mask emails
        sanitized_text = self.email_pattern.sub('[EMAIL MASKED]', sanitized_text)

        # Mask phone numbers
        sanitized_text = self.phone_pattern.sub('[PHONE MASKED]', sanitized_text)

        # Masking names and addresses reliably requires NLP (e.g., spaCy).
        # We'll use placeholder logic for demonstrating the MCP guard rules.

        # Simulated address masking (e.g. looking for common address keywords)
        address_keywords = ['Street', 'St', 'Avenue', 'Ave', 'Boulevard', 'Blvd', 'Road', 'Rd', 'Lane', 'Ln']
        for kw in address_keywords:
            # Mask the line containing the address keyword roughly
            sanitized_text = re.sub(rf'^.*?\b{kw}\b.*$', '[ADDRESS MASKED]', sanitized_text, flags=re.MULTILINE)

        # Simulated name masking
        # Assuming name is often at the top of the CV text or formatted in a specific way.
        # Here we just provide a function definition to comply with rules.
        # In practice: run NER on sanitized_text and replace PERSON labels.
        sanitized_text = self._mask_names_dummy(sanitized_text)

        return sanitized_text

    def _mask_names_dummy(self, text: str) -> str:
        # Dummy masking for demonstration.
        # This replaces lines that look like a standalone name (2-3 capitalized words).
        name_pattern = re.compile(r'^[A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?$', re.MULTILINE)
        return name_pattern.sub('[NAME MASKED]', text)
