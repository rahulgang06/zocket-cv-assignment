from flask import Flask, request, jsonify, send_file, render_template_string
from roboflow import Roboflow
from flask_cors import CORS
import tempfile
import os

app = Flask(__name__)
CORS(app)

rf = Roboflow(api_key="7iMfivyXV2bmm6U6rzGm")
project_name = "coco-dataset-vdnr1"
model_version = 11
model = rf.workspace().project(project_name).version(model_version).model


@app.route('/')
def upload_form():
    return render_template_string('''
        <!doctype html>
        <title>Upload an Image</title>
        <h1>Upload an Image for Prediction</h1>
        <form method="post" action="/predict" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <input type="submit" value="Upload">
        </form>
    ''')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file to a temporary location
        temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        file.save(temp_input_file.name)
        temp_input_file.close()

        # Make predictions using the model
        predictions = model.predict(temp_input_file.name)

        # Save the predictions to another temporary file
        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        predictions.save(temp_output_file.name)

        # Send the resulting file to the client
        return send_file(temp_output_file.name, as_attachment=True, download_name='prediction.jpg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, jsonify, send_file
# from roboflow import Roboflow
# from flask_cors import CORS
# import tempfile
# import os
# 
# app = Flask(__name__)
# CORS(app)
# 
# rf = Roboflow(api_key="W9VikDr39oibIVgg5UOS")
# project_name = "coco-dataset-vdnr1"
# model_version = 11
# model = rf.workspace().project(project_name).version(model_version).model
# 
# 
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"}), 400
# 
#         file = request.files['file']
# 
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400
# 
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
# 
#         predictions = model.predict(file)
#         predictions.save(temp_file.name)
# 
#         print("Here")
#         return send_file(temp_file.name, as_attachment=True, download_name='prediction.jpg')
# 
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
# 
# 
# if __name__ == '__main__':
#     app.run(debug=True)
