import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json


BASE_DIR = os.path.realpath(__file__)


class RequestHandler(BaseHTTPRequestHandler):
    def answer_json(self, status: int, message: str, data: dict):
        response = bytes(json.dumps(data), "utf8")
        self.send_response(status, message)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Content-length", len(response))
        self.end_headers()
        self.wfile.write(response)
    
    def answer_html(self, status: int, message: str, path: str):
        with open(path, 'r') as html:
            html.seek(0, os.SEEK_END)
            size = html.tell()
            self.send_response(status, message)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Content-length", html.tell())
            self.end_headers()
            html.seek(0)
            buf = html.read(65535)
            while buf:
                self.wfile.write(bytes(buf, "utf8"))
                buf = html.read(65535)
    
    def answer_plain(self, status: int, message: str, data: str):
        response = bytes(data, "utf8")
        self.send_response(status, message)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-length", len(response))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self):
        if(self.path in ("/", "/index.html")):
            self.answer_html(200, "OK", os.path.join(os.path.abspath(os.curdir), 'server/res/index.html'))
        elif(self.path == "/mindist"):
            self.answer_json(200, "OK", {"a":1, "b":2})
        else:
            self.answer_html(404, "Not Found", os.path.join(os.path.abspath(os.curdir), 'server/res/not_found.html'))

    def do_POST(self):
        print(self.path, self.rfile.read())
        self.answer_plain(200, "OK", "NO")

def create_server(port: int) -> HTTPServer:
    address = ('', port)
    httpd = HTTPServer(address, RequestHandler)
    httpd.serve_forever()