#!/usr/bin/env python3
"""
Run the Kerala Ayurveda RAG API Server.

Usage:
    python run_api.py [--port PORT] [--host HOST] [--reload]
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Kerala Ayurveda RAG API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         🌿 Kerala Ayurveda RAG API Server                    ║
╠══════════════════════════════════════════════════════════════╣
║  Starting server...                                          ║
║  Host: {args.host:<54} ║
║  Port: {args.port:<54} ║
║  Reload: {str(args.reload):<52} ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║  • POST /generate   - Generate content                       ║
║  • POST /query      - Q&A queries                            ║
║  • POST /safety-check - Check contraindications              ║
║  • POST /evaluate   - Evaluate responses                     ║
║  • GET  /health     - Health check                           ║
║  • GET  /stats      - System statistics                      ║
║  • GET  /docs       - Swagger documentation                  ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
