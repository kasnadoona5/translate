FROM python:3.11-slim

# System dependencies for reportlab, arabic-reshaper, and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: uncomment for OCR support
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     tesseract-ocr \
#     tesseract-ocr-fas \
#     ghostscript \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY glossary/ glossary/
COPY config.example.toml ./

# Install the package
RUN pip install --no-cache-dir .

# Create directories for persistent data
RUN mkdir -p /app/jobs /app/output

# Expose web UI port
EXPOSE 8080

# Default command: run the web UI
CMD ["tarjomeh", "serve", "--host", "0.0.0.0", "--port", "8080"]
