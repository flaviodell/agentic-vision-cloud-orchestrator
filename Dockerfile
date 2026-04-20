# Lambda-compatible base image
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum install -y libgomp && yum clean all

# Pin numpy first to avoid build-from-source issues
RUN pip install --no-cache-dir "numpy<2.0"

# Install PyTorch CPU-only — increased timeout for large downloads
RUN pip install --no-cache-dir --timeout 120 torch==2.4.1+cpu torchvision==0.19.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY cv_service/requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

# Install Mangum (Lambda adapter for FastAPI)
RUN pip install --no-cache-dir "mangum>=0.17.0"

# Copy CV service application code
COPY cv_service/app/ ./app/

# Lambda handler
CMD ["app.main.handler"]