#!/bin/bash
# Pi-Whispr Development Setup Script

set -e

echo "🎤 Pi-Whispr Development Setup"
echo "==============================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create environment file
if [ ! -f .env ]; then
    echo "📄 Creating .env file from development template..."
    cp config/development.env .env
    echo "✅ Created .env file (edit with your Pi's IP address when ready)"
else
    echo "📄 .env file already exists"
fi

# Build development container
echo "🐳 Building development container..."
docker-compose build dev-server

# Test the setup
echo "🧪 Testing development setup..."
docker-compose up -d dev-server

# Wait a moment for the server to start
sleep 3

# Check if server is responding
if curl -f http://localhost:8765/health &> /dev/null; then
    echo "✅ Development server is running successfully!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Install client dependencies: pip3 install -r requirements/client.txt"
    echo "   2. Update .env with your Pi's IP address when ready for Pi deployment"
    echo "   3. Run client: python client/speech_client.py (when implemented)"
    echo ""
    echo "📖 Development workflow:"
    echo "   - Start dev server: docker-compose up dev-server"
    echo "   - View logs: docker-compose logs -f dev-server"
    echo "   - Stop server: docker-compose down"
else
    echo "⚠️  Development server started but health check failed"
    echo "   Check logs with: docker-compose logs dev-server"
fi

# Clean up test container
docker-compose down

echo "�� Setup complete!" 