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

# Install declared runtime dependencies in a source-independent layer. Code-only
# releases can then reuse this large layer on the small production VPS.
COPY pyproject.toml README.md LICENSE ./
RUN python -c "import subprocess,sys,tomllib; p=tomllib.load(open('pyproject.toml','rb')); subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir',*p['project']['dependencies']])"

# Copy project files
COPY src/ src/
COPY glossary/ glossary/
COPY config.example.toml ./

# Dependencies are already present in the stable layer above.
RUN pip install --no-cache-dir --no-deps .

# Create directories for persistent data
RUN mkdir -p /app/jobs /app/output

# Expose web UI port
EXPOSE 8080

# Default command: run the web UI
CMD ["tarjomeh", "serve", "--host", "0.0.0.0", "--port", "8080"]
