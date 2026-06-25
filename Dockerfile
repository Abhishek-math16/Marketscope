FROM python:3.11-slim

# Hugging Face requires a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files
COPY --chown=user . /app

# Create folders that the app writes to at runtime
RUN mkdir -p models datasets data static

EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "app:app"]
