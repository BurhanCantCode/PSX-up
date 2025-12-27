"""
Health Check API Endpoint
Simple endpoint to verify the API is running.
"""

import json
from http.server import BaseHTTPRequestHandler
from datetime import datetime


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        result = {
            'status': 'healthy',
            'service': 'psx-prediction-api',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0'
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))

