from flask import Flask

from generation import get_text


app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return get_text()


if __name__ == '__main__':
    app.run()
