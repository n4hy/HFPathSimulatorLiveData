# HF Path Simulator - Web Dashboard Docker image
# Multi-stage build for React/TypeScript

# ==============================================================================
# Stage 1: Development (for docker-compose.dev.yml)
# ==============================================================================
FROM node:20-alpine AS development

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source (will be mounted in dev mode)
COPY . .

# Expose Vite dev server port
EXPOSE 3000

# Default command for development
CMD ["npm", "start"]

# ==============================================================================
# Stage 2: Build
# ==============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

# Build for production
RUN npm run build

# ==============================================================================
# Stage 3: Production (nginx)
# ==============================================================================
FROM nginx:alpine AS production

# Labels
LABEL maintainer="HFPathSim Contributors"
LABEL org.opencontainers.image.title="HF Path Simulator Web Dashboard"
LABEL org.opencontainers.image.description="React-based web dashboard for HF Path Simulator"
LABEL org.opencontainers.image.version="0.1.0"

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:3000/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
