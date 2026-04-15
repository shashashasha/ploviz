import http.server
import ssl
import os

PORT = 8443
base = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=base, **kwargs)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(os.path.join(base, "../certs/cert.pem"), os.path.join(base, "../certs/key.pem"))

with http.server.HTTPServer(("0.0.0.0", PORT), Handler) as httpd:
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    print(f"Serving https://192.168.68.58:{PORT}")
    httpd.serve_forever()
