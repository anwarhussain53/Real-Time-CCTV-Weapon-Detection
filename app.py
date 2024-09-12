from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Handle file and perform detection here
    # For demonstration purposes, we return a mock response
    return jsonify({'result': 'No firearm detected'})

if __name__ == '__main__':
    app.run(debug=True)
