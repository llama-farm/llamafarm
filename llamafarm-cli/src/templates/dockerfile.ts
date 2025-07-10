
 export function createDockerfile(device: string, model: string, config: any): string {
    const baseImage = device === 'jetson' ? 'nvcr.io/nvidia/l4t-ml:r35.2.1-py3' : 'ubuntu:22.04';
    
    return `FROM ${baseImage}
  
  # Install system dependencies
  RUN apt-get update && apt-get install -y \\
      curl \\
      git \\
      python3 \\
      python3-pip \\
      nodejs \\
      npm \\
      && rm -rf /var/lib/apt/lists/*
  
  # Install Ollama
  RUN curl -fsSL https://ollama.ai/install.sh | sh
  
  # Set working directory
  WORKDIR /app
  
  # Copy application files
  COPY . .
  
  # Install Node dependencies
  RUN npm install
  
  # Install Python dependencies for vector DB
  RUN pip3 install chromadb
  
  # Pull the model
  RUN ollama pull ${model}
  
  # Expose ports
  EXPOSE ${config.port || 8080}
  EXPOSE ${(config.port || 8080) + 1}
  
  # Start script
  CMD ["./start.sh"]
  `;
  }