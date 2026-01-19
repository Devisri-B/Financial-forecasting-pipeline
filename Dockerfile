# Use AWS Lambda base image for Python 3.10
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements separately to cache build steps
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy source code and models
COPY src/ ${LAMBDA_TASK_ROOT}/src
COPY models/ ${LAMBDA_TASK_ROOT}/models
COPY config/ ${LAMBDA_TASK_ROOT}/config

# Set the CMD to your handler (file_name.variable_name)
CMD [ "src.app.handler" ]