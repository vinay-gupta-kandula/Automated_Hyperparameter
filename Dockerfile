FROM python:3.9

WORKDIR /app

# ---- Install system dependencies + Chrome ----
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    fonts-liberation \
    libnss3 \
    libatk-bridge2.0-0 \
    libxss1 \
    libasound2 \
    libgbm1 \
    libgtk-3-0 \
    libu2f-udev \
    libvulkan1 \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install kaleido plotly

# ---- Copy project files ----
COPY src ./src
COPY notebooks ./notebooks
COPY data ./data

# ---- Outputs directory ----
RUN mkdir -p /app/outputs

# ---- Run pipeline ----
CMD ["python", "src/optimize.py"]
