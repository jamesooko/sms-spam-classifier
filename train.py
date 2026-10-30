"""Train and evaluate a local SMS spam classifier. No API keys required."""
from pathlib import Path
import argparse,csv,hashlib,json,re,unicodedata,urllib.request,zipfile,platform
import numpy as np
import sklearn,joblib
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score,roc_auc_score,average_precision_score,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
SOURCE='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
def normalize(text):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).casefold()).strip()
def load_data():
 folder=ROOT;folder.mkdir(exist_ok=True);archive=folder/'source.zip'
 if not archive.exists():urllib.request.urlretrieve(SOURCE,archive)
 with zipfile.ZipFile(archive) as z:lines=z.read('SMSSpamCollection').decode('utf-8').splitlines()
 rows=[]
 for i,line in enumerate(lines):
  if not line.strip():continue
  label,text=line.split('\t',1)
  if label not in {'ham','spam'} or not text.strip():raise ValueError('Unexpected label or empty message')
  rows.append((i,label,text,normalize(text)))
 labels={}
 for _,label,_,key in rows:labels.setdefault(key,set()).add(label)
 conflicts={key for key,v in labels.items() if len(v)>1};seen=set();clean=[];duplicates=0;excluded=0
 for row in rows:
  if row[3] in conflicts:excluded+=1;continue
  if row[3] in seen:duplicates+=1;continue
  seen.add(row[3]);clean.append(row)
 return clean,{'source_url':SOURCE,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'raw_rows':len(rows),'conflicting_text_groups':len(conflicts),'conflict_rows_excluded':excluded,'duplicate_rows_removed':duplicates,'clean_rows':len(clean)}
def main():
 rows,checks=load_data();out=ROOT;out.mkdir(exist_ok=True)
 texts=np.array([r[2] for r in rows]);y=np.array([r[1]=='spam' for r in rows],dtype=int)
 train,test=train_test_split(np.arange(len(rows)),test_size=.2,stratify=y,random_state=42)
 assert not ({rows[i][3] for i in train}&{rows[i][3] for i in test})
 checks.update({'train_rows':len(train),'test_rows':len(test),'train_spam':int(y[train].sum()),'test_spam':int(y[test].sum()),'normalized_text_overlap':0})
 cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
 specs={'logistic_regression':(LogisticRegression(max_iter=2000,random_state=42),{'classifier__C':[.5,2,8]}),'naive_bayes':(MultinomialNB(),{'classifier__alpha':[.1,.5,1]})}
 fitted={};cv_rows=[]
 for name,(estimator,grid) in specs.items():
  pipe=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True)),('classifier',estimator)])
  search=GridSearchCV(pipe,grid,scoring='f1',cv=cv,n_jobs=1,return_train_score=False).fit(texts[train],y[train]);fitted[name]=search.best_estimator_
  cv_rows.append({'model':name,'training_cv_spam_f1':float(search.best_score_),'parameters':search.best_params_})
 selected=max(cv_rows,key=lambda r:r['training_cv_spam_f1'])['model']
 reports=[];predictions=[]
 for name,model in fitted.items():
  probability=model.predict_proba(texts[test])[:,1];pred=(probability>=.5).astype(int);p,r,f,_=precision_recall_fscore_support(y[test],pred,average='binary',zero_division=0);cm=confusion_matrix(y[test],pred,labels=[0,1])
  reports.append({'model':name,'selected_by_training_cv':name==selected,'accuracy':accuracy_score(y[test],pred),'spam_precision':p,'spam_recall':r,'spam_f1':f,'roc_auc':roc_auc_score(y[test],probability),'average_precision':average_precision_score(y[test],probability),'tn':int(cm[0,0]),'fp':int(cm[0,1]),'fn':int(cm[1,0]),'tp':int(cm[1,1])})
  for idx,actual,guess,score in zip(test,y[test],pred,probability):predictions.append({'model':name,'source_row':rows[idx][0],'text_sha256':hashlib.sha256(rows[idx][3].encode()).hexdigest(),'actual':int(actual),'predicted':int(guess),'spam_score':float(score)})
 baseline={'accuracy':float(np.mean(y[test]==0)),'spam_recall':0,'spam_f1':0}
 result={'data_checks':checks,'seed':42,'test_fraction':.2,'cv_folds':5,'selection_metric':'training cross-validation spam F1','decision_threshold':.5,'selected_model':selected,'cross_validation':cv_rows,'test_metrics':reports,'always_ham_baseline':baseline,'versions':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,'numpy':np.__version__}}
 (out/'metrics.json').write_text(json.dumps(result,indent=2))
 with (out/'test_predictions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=predictions[0].keys());w.writeheader();w.writerows(predictions)
 with (out/'split_membership.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['source_row','split','label','normalized_text_sha256'])
  for split,indices in [('train',train),('test',test)]:
   for i in indices:w.writerow([rows[i][0],split,rows[i][1],hashlib.sha256(rows[i][3].encode()).hexdigest()])
 joblib.dump(fitted[selected],out/'spam_model.joblib')
 plt.rcParams.update({'figure.facecolor':'#061B2D','axes.facecolor':'#061B2D','text.color':'#F4FAFC','axes.labelcolor':'#F4FAFC','xtick.color':'#A8BDCC','ytick.color':'#A8BDCC','axes.edgecolor':'#A8BDCC','font.size':12})
 model=fitted[selected];pred=model.predict(texts[test]);cm=confusion_matrix(y[test],pred,labels=[0,1]);scores=model.predict_proba(texts[test])[:,1]
 fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained');a=axes[0];a.imshow(cm,cmap='Blues');a.set(xticks=[0,1],yticks=[0,1],xticklabels=['Ham','Spam'],yticklabels=['Ham','Spam'],xlabel='Predicted label',ylabel='Actual label',title=f'Held-out messages: {len(test):,}')
 for i in range(2):
  for j in range(2):a.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=23,color='white' if cm[i,j]>cm.max()/2 else '#061B2D')
 p,r,_=precision_recall_curve(y[test],scores);a=axes[1];a.plot(r,p,color='#16D5D4',linewidth=2);a.axhline(y[test].mean(),color='#A8BDCC',linestyle='--',label='Spam prevalence');a.set(xlim=(0,1),ylim=(0,1.03),xlabel='Spam recall',ylabel='Spam precision',title='Precision-recall curve');a.legend(facecolor='#061B2D',labelcolor='white');fig.suptitle('SMS spam classification | '+selected.replace('_',' '),fontsize=19);fig.savefig(out/'evaluation.png',dpi=180);plt.close(fig)
 # Independent count arithmetic from saved predictions, not metric helpers.
 chosen=[p for p in predictions if p['model']==selected];tp=sum(p['actual']==1 and p['predicted']==1 for p in chosen);fp=sum(p['actual']==0 and p['predicted']==1 for p in chosen);fn=sum(p['actual']==1 and p['predicted']==0 for p in chosen)
 report=next(r for r in reports if r['model']==selected);assert abs(tp/(tp+fp)-report['spam_precision'])<1e-12;assert abs(tp/(tp+fn)-report['spam_recall'])<1e-12;assert tp+fp+fn+report['tn']==len(test)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
