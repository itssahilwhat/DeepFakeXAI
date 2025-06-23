import os
from flask import Flask, request, render_template
import requests

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        files = {"file": (file.filename, file.stream, file.content_type)}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)
        data = response.json()
        return render_template("index.html", result=data)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
