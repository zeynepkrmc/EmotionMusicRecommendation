from flask import Flask, render_template, Response, jsonify, session
from camera import *
from Spotipy import get_emotion_recommendations


app = Flask(__name__)

app.secret_key = 'mysecretkey114'

camera = VideoCamera()

@app.route('/')
def index():
    recommendations = session.get('recommendations', [])
    #print(df1.to_json(orient='records'))
    return render_template('index.html', recommendations=recommendations)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    global show_text
    emotion_id = show_text[0]  # En son algılanan yüz ifadesi
    print(f"Detected emotion: {emotion_id}")
    
    # Fetch recommendations from the function
    recommendations = get_emotion_recommendations(emotion_id)
    session['recommendations'] = recommendations  # Sakla
    print(f"Recommendations: {recommendations}")
    return jsonify(recommendations)

if __name__ == '__main__':
    app.debug = True
    app.run()
