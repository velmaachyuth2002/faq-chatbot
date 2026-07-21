# Start from a lightweight, official Python 3.10 image — matches the
# Python version we've been developing against, without the extra bulk of
# a full desktop-style Linux image.
FROM python:3.10-slim

# All following commands run inside this folder within the container.
WORKDIR /app

# Copy just the dependency list first, and install dependencies, before
# copying the rest of our code. Docker caches each step — this way, if you
# only change your Python code later (not your dependencies), Docker can
# reuse the cached "install dependencies" step instead of redoing it,
# making rebuilds much faster.
COPY requirements.txt .
# --extra-index-url points pip at PyTorch's CPU-only builds, in addition
# to the normal package index. sentence-transformers depends on PyTorch,
# and by default pip installs PyTorch's full GPU-capable build (several
# GB of NVIDIA CUDA libraries) even though this container has no GPU to
# use it and we're only running lightweight CPU calculations. This flag
# makes pip prefer the much smaller CPU-only build instead.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Now copy the actual application code and data into the image.
COPY faq_engine.py api.py faqs.json ./

# Documents that this container listens on port 8000. This line is
# informational — it doesn't actually publish the port (that happens
# with `docker run -p`), but it's good practice to declare it.
EXPOSE 8000

# The command that runs when a container starts from this image.
# --host 0.0.0.0 (instead of 127.0.0.1) is required inside Docker so the
# server accepts connections from outside the container, not just from
# within it.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
