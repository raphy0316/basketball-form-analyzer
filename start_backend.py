#!/usr/bin/env python3
"""
Startup script for the Basketball Form Analyzer Backend API
"""

import uvicorn
import os
import sys

def main():
    """Start the FastAPI backend server"""
    
    # Check if required dependencies are installed
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Set the host and port
    host = "0.0.0.0"  # Allow external connections
    port = 8000
    
    print(f"🚀 Starting Basketball Form Analyzer Backend API...")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📱 Mobile app should connect to: http://192.168.0.165:{port}")
    print(f"🔗 API Documentation: http://{host}:{port}/docs")
    print(f"🏥 Health Check: http://{host}:{port}/health")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
