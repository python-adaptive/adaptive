# Start from the base image
FROM mcr.microsoft.com/devcontainers/python

# Install packages
RUN pipx install ruff

# Copy the project's source code
COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Install the project's dependencies
RUN pip install -e ".[other,notebook,testing,dev]"

# Install pre-commit hooks
RUN pre-commit install
