# Build stage - using bullseye instead of alpine for better ARM64 support
FROM node:20-bullseye-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies with a workaround for the rollup ARM64 issue
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Workaround for rollup ARM64 dependency issue
RUN rm -rf node_modules package-lock.json && \
    npm install --legacy-peer-deps && \
    npm run build

# Production stage
FROM nginx:alpine

# Install jq for environment variable processing
RUN apk add --no-cache jq

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration for SPA support
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Set working directory to where the built files are
WORKDIR /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Use custom entrypoint that processes environment variables
ENTRYPOINT ["/docker-entrypoint.sh"]

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
