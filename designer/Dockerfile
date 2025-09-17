ARG BUILDPLATFORM
ARG TARGETPLATFORM
# Build stage - use Alpine to avoid dpkg/qemu issues on multi-arch
FROM --platform=$BUILDPLATFORM node:20-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 make g++

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Disable husky/lifecycle scripts during container installs
ENV HUSKY=0

# Install dependencies (allow scripts so rollup can install its native binary)
RUN npm ci --legacy-peer-deps || npm install --legacy-peer-deps

# Copy source code
COPY . .

# Workaround for rollup optional native binary on platform images
# npm recommends reinstalling without lock after copying sources
RUN npm cache clean --force \
    && rm -rf node_modules package-lock.json \
    && npm install --legacy-peer-deps

# Build
RUN npm run build

# Production stage (target platform)
FROM --platform=$TARGETPLATFORM nginx:alpine

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
