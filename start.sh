#!/bin/bash
# MOH Tender OCR Web Application Startup Script

echo "🚀 Starting MOH Tender OCR Web Application..."
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$SCRIPT_DIR/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"
echo ""

# Setup Python virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r "$BACKEND_DIR/requirements.txt"

# Install Node dependencies if needed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    cd "$FRONTEND_DIR"
    npm install
    cd "$SCRIPT_DIR"
fi

echo ""
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Start backend server in background
echo "🔧 Starting backend server on http://localhost:8000..."
cd "$BACKEND_DIR"
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait for backend to start
sleep 3

# Start frontend dev server
echo "🎨 Starting frontend server on http://localhost:5173..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  MOH Tender OCR Web Application is running!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "  📊 Frontend:  http://localhost:5173"
echo "  🔧 Backend:   http://localhost:8000"
echo "  📚 API Docs:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop all servers"
echo ""

# Handle shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ Servers stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
