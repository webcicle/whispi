#!/bin/bash
# Pi-Whispr Testing Workflow
# Runs complete testing suite before Pi deployment

set -e  # Exit on any error

echo "🧪 Pi-Whispr Testing Workflow"
echo "============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Parse command line arguments first (before environment validation)
QUICK_TEST=false
FULL_TEST=false
CLIENT_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_TEST=true
            shift
            ;;
        --full)
            FULL_TEST=true
            shift
            ;;
        --client)
            CLIENT_TEST=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--quick|--full|--client]"
            echo "  --quick: Run basic connection tests only"
            echo "  --full:  Run comprehensive test suite"
            echo "  --client: Run client functionality tests"
            echo "  --help:  Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Usage: $0 [--quick|--full|--client]"
            echo "Use --help for more information"
            exit 1
            ;;
    esac
done

# Default to full test if no option specified
if [[ "$QUICK_TEST" == false && "$FULL_TEST" == false && "$CLIENT_TEST" == false ]]; then
    FULL_TEST=true
fi

# Function to detect and activate virtual environment
activate_virtual_environment() {
    # Check if virtual environment exists
    if [[ -d "venv" && -f "venv/bin/activate" ]]; then
        print_step "🐍 Found virtual environment, activating..."
        source venv/bin/activate
        print_success "Virtual environment activated"
        
        # Export the activated environment to subprocesses
        export VIRTUAL_ENV="$(pwd)/venv"
        export PATH="$VIRTUAL_ENV/bin:$PATH"
    elif [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Already in virtual environment: $VIRTUAL_ENV"
    else
        print_warning "No virtual environment found. Using system Python."
        print_warning "Consider creating a virtual environment with: python3 -m venv venv"
    fi
}

# Function to detect Python and pip commands
detect_python_commands() {
    # First activate virtual environment if available
    activate_virtual_environment
    
    # Try to find Python command (prefer virtual env versions)
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        print_error "Neither python nor python3 found. Please install Python."
        exit 1
    fi
    
    # Try to find pip command (prefer python -m pip for virtual env compatibility)
    if $PYTHON_CMD -m pip --version &> /dev/null 2>&1; then
        PIP_CMD="$PYTHON_CMD -m pip"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
    elif command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    else
        print_error "pip not found. Please install pip."
        exit 1
    fi
    
    # Show which Python version we're using
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    print_success "Using $PYTHON_CMD ($PYTHON_VERSION) and $PIP_CMD"
}

# Function to detect docker-compose command
detect_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    elif command -v docker &> /dev/null && docker compose version &> /dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        print_error "docker-compose is not available. Please install Docker Compose."
        exit 1
    fi
    print_success "Using $DOCKER_COMPOSE_CMD"
}

# Detect Python and pip commands
print_step "🐍 Detecting Python environment..."
detect_python_commands

# Run comprehensive environment validation
print_step "🔍 Validating testing environment..."
if ! $PYTHON_CMD validate_environment.py; then
    print_error "Environment validation failed. Please fix the issues above."
    exit 1
fi
print_success "Environment validation passed"

# Detect docker-compose command
print_step "🐚 Detecting docker-compose..."
detect_docker_compose

# Install testing dependencies if needed
print_step "📦 Installing/updating testing dependencies..."
if $PIP_CMD install -r requirements_test.txt; then
    print_success "Dependencies ready"
else
    print_error "Failed to install dependencies. Try running manually:"
    print_error "  $PIP_CMD install -r requirements_test.txt"
    exit 1
fi

# Clean up any existing containers
print_step "🧹 Cleaning up previous containers..."
$DOCKER_COMPOSE_CMD down -v 2>/dev/null || true
print_success "Cleanup complete"

# Function to wait for container to be ready
wait_for_container() {
    local container_name=$1
    local max_wait=${2:-30}
    local count=0
    
    print_step "⏳ Waiting for $container_name to be ready..."
    
    while [[ $count -lt $max_wait ]]; do
        if $DOCKER_COMPOSE_CMD ps $container_name | grep -q "Up"; then
            # Additional check - try to connect
            if $PYTHON_CMD test_connection.py --server-url "ws://localhost:8765" >/dev/null 2>&1; then
                print_success "$container_name is ready"
                return 0
            fi
        fi
        
        sleep 2
        ((count++))
        
        if [[ $((count % 5)) -eq 0 ]]; then
            print_warning "Still waiting for $container_name... ($count/$max_wait)"
        fi
    done
    
    print_error "$container_name failed to start within ${max_wait} seconds"
    print_error "Container logs:"
    $DOCKER_COMPOSE_CMD logs $container_name
    return 1
}

if [[ "$QUICK_TEST" == true ]]; then
    print_step "🚀 Running Quick Connection Tests..."
    
    # Start mock server
    print_step "Starting mock server..."
    if ! $DOCKER_COMPOSE_CMD up -d mock-server; then
        print_error "Failed to start mock server"
        exit 1
    fi
    
    # Wait for server to be ready
    if ! wait_for_container "mock-server"; then
        exit 1
    fi
    
    # Run connection tests (only mock-server for quick test)
    print_step "Running connection tests..."
    if $PYTHON_CMD test_connection.py --server-url "ws://localhost:8765"; then
        print_success "Quick tests completed!"
    else
        print_error "Connection tests failed"
        $DOCKER_COMPOSE_CMD logs mock-server
        exit 1
    fi
    
elif [[ "$CLIENT_TEST" == true ]]; then
    print_step "🎯 Running Client Functionality Tests..."
    
    # Start mock server
    print_step "Starting mock server..."
    if ! $DOCKER_COMPOSE_CMD up -d mock-server; then
        print_error "Failed to start mock server"
        exit 1
    fi
    
    # Wait for server to be ready
    if ! wait_for_container "mock-server"; then
        exit 1
    fi
    
    # Run client tests
    print_step "Running client tests..."
    if $PYTHON_CMD client_runner.py --mode mock --duration 20; then
        print_success "Client tests completed!"
    else
        print_error "Client tests failed"
        $DOCKER_COMPOSE_CMD logs mock-server
        exit 1
    fi
    
elif [[ "$FULL_TEST" == true ]]; then
    print_step "🔥 Running Full Test Suite..."
    
    # Run comprehensive test environment
    if $PYTHON_CMD test_environment.py; then
        print_success "All tests passed! 🎉"
        echo ""
        echo "Your Pi-Whispr system is ready for Pi deployment!"
        echo ""
        echo "Next steps:"
        echo "1. Copy the code to your Raspberry Pi"
        echo "2. Run: $DOCKER_COMPOSE_CMD up pi-server"
        echo "3. Test with your Pi client"
    else
        print_error "Some tests failed. Please check the output above."
        
        # Show container logs for debugging
        print_step "📋 Container logs for debugging:"
        for service in mock-server mock-pi5 mock-stress; do
            if $DOCKER_COMPOSE_CMD ps $service 2>/dev/null | grep -q "Up"; then
                echo "--- $service logs ---"
                $DOCKER_COMPOSE_CMD logs --tail=20 $service
            fi
        done
        
        exit 1
    fi
fi

# Cleanup
print_step "🧹 Cleaning up test containers..."
$DOCKER_COMPOSE_CMD down -v
print_success "Cleanup complete"

echo ""
print_success "Testing workflow completed!" 