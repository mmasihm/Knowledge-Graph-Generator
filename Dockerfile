FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/share/nltk_data

# OpenCV, pulled in by unstructured[pdf], links against these system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code changes don't invalidate this (slow) layer.
# Torch comes from the CPU-only index: the default CUDA wheels add several GB we never use.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir uv==0.11.32 \
    && uv pip install --system --no-cache torch==2.15.0 torchvision==0.29.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system --no-cache -r requirements.txt -r requirements-dev.txt

# Unstructured classifies text with NLTK; without this data partition_pdf returns nothing.
RUN python -c "import nltk; [nltk.download(p, download_dir='$NLTK_DATA', quiet=True) for p in ('punkt_tab', 'averaged_perceptron_tagger_eng')]"

COPY kg_generator ./kg_generator
COPY tests ./tests
COPY pytest.ini ./

RUN useradd --create-home app && mkdir -p state logs && chown app state logs
USER app

CMD ["python", "-m", "kg_generator.main"]
