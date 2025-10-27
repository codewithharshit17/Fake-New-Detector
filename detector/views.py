from django.shortcuts import render
from .ml_model import FakeNewsDetector
import time

MODEL_STATS = {
    "accuracy": 97.32,
    "real_precision": 0.98,
    "real_recall": 0.96,
    "real_f1": 0.97,
    "fake_precision": 0.96,
    "fake_recall": 0.97,
    "fake_f1": 0.96,
    "train_size": 40000,
    "real_count": 22000,
    "fake_count": 18000
}

def home(request):
    return render(request, 'detector/home.html')

def detect_news(request):
    result, confidence, news_text = "", 0, ""
    input_words = 0
    prediction_time = 0
    session_real_count = request.session.get('session_real_count', 0)
    session_fake_count = request.session.get('session_fake_count', 0)
    confidence_remaining = 100
    if request.method == 'POST':
        news_text = request.POST.get('news_text', "").strip()
        input_words = len(news_text.split())
        start = time.time()
        detector = FakeNewsDetector()
        detector.load_model()
        prediction, confidence = detector.predict(news_text)
        prediction_time = round(time.time() - start, 2)
        result = 'This news appears to be FAKE' if prediction == 'FAKE' else 'This news appears to be REAL'
        confidence_remaining = 100 - float(confidence) if confidence is not None else 100
        if prediction == 'REAL':
            session_real_count += 1
        else:
            session_fake_count += 1
        request.session['session_real_count'] = session_real_count
        request.session['session_fake_count'] = session_fake_count
    return render(request, 'detector/result.html', {
        'result': result,
        'confidence': float(confidence),
        'confidence_remaining': float(confidence_remaining),
        'news_text': news_text,
        'input_words': input_words,
        'prediction_time': prediction_time,
        'model_stats': MODEL_STATS,
        'session_real_count': session_real_count,
        'session_fake_count': session_fake_count,
    })


# If you want a stats page (for training graphs)
def stats(request):
    import os, glob
    plot_folder = os.path.join('media', 'plots')
    all_plots = []
    if os.path.exists(plot_folder):
        all_plots = sorted([os.path.basename(f) for f in glob.glob(f"{plot_folder}/*.png")])
    return render(request, 'detector/stats.html', {'plots': all_plots})
