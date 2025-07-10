
export function createDockerfile(device: string, model: string, config: any): string {
  const baseImage = device === 'raspberry-pi' ? 'arm64v8/node:18-slim' : 'node:18-slim';
  
  return `FROM ${baseImage}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    git \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Node.js dependencies
RUN npm install --production

# Install Python dependencies for vector DB
RUN pip3 install chromadb

# Environment variables
ENV NODE_ENV=production
ENV MODEL=${model}
ENV DEVICE=${device}
ENV PORT=${config.port || 8080}

# Expose port
EXPOSE ${config.port || 8080}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD node -e "require('http').get('http://localhost:${config.port || 8080}/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

# Start command
CMD ["./start.sh"]
`;
}