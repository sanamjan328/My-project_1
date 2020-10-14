# coding=utf-8

import os
from model import app
__author__ = 'sanam'

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '4555'))
    except ValueError:
        PORT = 4555
    app.run(HOST, PORT,debug=True)
