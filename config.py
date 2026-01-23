"""
Central configuration for the Finance Guidance Extraction project.

This module centralizes configuration values that were previously hardcoded
across multiple files. Load from environment variables where appropriate.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

# Contact email for User-Agent strings (required by SEC.gov and other APIs)
# Can be overridden via environment variable
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "your-email@example.com")

# User-Agent string for web requests
USER_AGENT = f"AgenticFinanceResearchBot/0.1 (contact: {CONTACT_EMAIL})"

# SEC-specific User-Agent (more formal, includes research purpose)
SEC_USER_AGENT = f"AgenticFinanceResearchBot/0.1 (Academic Research; contact: {CONTACT_EMAIL})"

# Timezone for timestamps
TIMEZONE = "Asia/Kuala_Lumpur"
