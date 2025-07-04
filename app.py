
from flask import Flask, render_template, request
import subprocess
import os
from green_signal_time import adjust_signal_time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_detection():
    try:
        # Get uploaded image
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.png')
            image_file.save(image_path)

            # Run detection with flag
            subprocess.run(['python', 'vehicle_detection.py','--from-flask'], check=True)

            # Read vehicle count
            with open("vehicle_count.txt", "r") as f:
                vehicle_count = int(f.read().strip())
            green_time = adjust_signal_time(vehicle_count)

            return render_template('result.html', count=vehicle_count, time=green_time)
        else:
            return "<h3>No image uploaded</h3>"

    except Exception as e:
        return f"<h3>Error occurred: {e}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
