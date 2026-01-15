# Use official Python image - specify amd64 for AWS compatibility
FROM --platform=linux/amd64 python:3.9

# Set working directory
WORKDIR /app

# Set Hugging Face cache directory
ENV HF_HOME=/app/.cache/huggingface

# Copy setup.py and requirements first
COPY setup.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models from Hugging Face
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='cyuangli/embedding-model', filename='embedding_model.keras', repo_type='model'); \
    hf_hub_download(repo_id='cyuangli/pca-data', filename='pca.joblib', repo_type='model'); \
    hf_hub_download(repo_id='cyuangli/recipe-faiss', filename='recipes.faiss', repo_type='model'); \
    hf_hub_download(repo_id='cyuangli/image-data', filename='image_paths.npy', repo_type='model'); \
    hf_hub_download(repo_id='cyuangli/WebEats-v3', filename='recipe_meta_topics.csv', repo_type='dataset')"

# Copy necessary folders
COPY app/ ./app/
COPY src/ ./src/

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]