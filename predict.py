"""Score a message locally with the model created by train.py."""
import argparse,json
from pathlib import Path
import joblib
p=argparse.ArgumentParser();p.add_argument('message',help='Message text, in quotes');args=p.parse_args()
if not args.message.strip():p.error('Message cannot be blank')
model=joblib.load(Path(__file__).resolve().parent/'spam_model.joblib')
score=float(model.predict_proba([args.message])[0,1]);print(json.dumps({'prediction':'spam' if score>=.5 else 'ham','model_spam_score':round(score,4),'note':'Educational model score, not a guarantee that a message is safe.'},indent=2))
