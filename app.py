import os,json,sqlite3,urllib.parse
from datetime import datetime,timezone
import numpy as np,pandas as pd,streamlit as st,plotly.graph_objects as go,yfinance as yf,feedparser
import requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from market_data import fetch_bluesky_posts,fetch_gdelt_news,fetch_reddit_posts,normalize_yahoo_news,point_in_time_news_feature
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,mean_absolute_error,precision_score,brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

load_dotenv()
st.set_page_config(page_title='Bharat Market AI Production',page_icon='📈',layout='wide')
DB=os.getenv('APP_DB_PATH','stock_ai.db'); AN=SentimentIntensityAnalyzer()
SECTORS={
'Information Technology':{'TCS':'TCS','Infosys':'INFY','HCL Tech':'HCLTECH','Wipro':'WIPRO'},
'Banking':{'HDFC Bank':'HDFCBANK','ICICI Bank':'ICICIBANK','SBI':'SBIN','Axis Bank':'AXISBANK'},
'FMCG':{'ITC':'ITC','HUL':'HINDUNILVR','Nestle India':'NESTLEIND'},
'Pharma':{'Sun Pharma':'SUNPHARMA','Cipla':'CIPLA','Dr Reddy':'DRREDDY'},
'Automobile':{'Tata Motors':'TATAMOTORS','Maruti':'MARUTI','M&M':'M&M'},
'Oil & Gas':{'Reliance':'RELIANCE','ONGC':'ONGC','IOC':'IOC'},
'Power':{'NTPC':'NTPC','Power Grid':'POWERGRID','Tata Power':'TATAPOWER'},
'Metals':{'Tata Steel':'TATASTEEL','Hindalco':'HINDALCO','JSW Steel':'JSWSTEEL'},
'Telecom':{'Bharti Airtel':'BHARTIARTL','Vodafone Idea':'IDEA'},
'Defense':{'HAL':'HAL','Bharat Electronics':'BEL'},
'Railway':{'IRCTC':'IRCTC','RVNL':'RVNL'},'Realty':{'DLF':'DLF','Godrej Properties':'GODREJPROP'},
'Financial Services':{'Bajaj Finance':'BAJFINANCE','Bajaj Finserv':'BAJAJFINSV'},
'Infrastructure':{'Larsen & Toubro':'LT','Adani Ports':'ADANIPORTS'}}
H={'1 Day':1,'3 Days':3,'7 Days':7,'15 Days':15,'30 Days':30,'3 Months':63,'1 Year':252}

def conn():
 c=sqlite3.connect(DB);c.row_factory=sqlite3.Row;return c
def init():
 c=conn();c.executescript('''CREATE TABLE IF NOT EXISTS news_history(id INTEGER PRIMARY KEY,symbol TEXT,captured_at TEXT,published_at TEXT,source TEXT,headline TEXT,url TEXT,sentiment REAL,relevance REAL,UNIQUE(symbol,headline,published_at));
 CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY,symbol TEXT,prediction_date TEXT,target_date TEXT,horizon INTEGER,latest_close REAL,predicted_close REAL,predicted_return REAL,signal TEXT,confidence REAL,bullish REAL,bearish REAL,low REAL,high REAL,market_condition TEXT,market_return REAL,sector_return REAL,news_score REAL,news_json TEXT,feature_json TEXT,model_version TEXT,actual_close REAL,actual_return REAL,percentage_error REAL,direction_correct INTEGER,evaluated_at TEXT,UNIQUE(symbol,prediction_date,horizon,model_version));
 CREATE TABLE IF NOT EXISTS model_registry(id INTEGER PRIMARY KEY,symbol TEXT,horizon INTEGER,version TEXT,trained_at TEXT,rows INTEGER,metrics_json TEXT,active INTEGER);''');c.commit();c.close()
init()
@st.cache_data(ttl=300,show_spinner=False)
def hist(ticker,period='5y'):
 d=yf.download(ticker,period=period,auto_adjust=True,progress=False,threads=False)
 if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
 if d.empty:raise ValueError('No data for '+ticker)
 return d.dropna()
def indicators(d):
 x=d.copy();c=x.Close;v=x.Volume;delta=c.diff();up=delta.clip(lower=0);dn=-delta.clip(upper=0);rs=up.ewm(alpha=1/14,adjust=False).mean()/dn.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
 x['RSI']=100-100/(1+rs);x['EMA20']=c.ewm(span=20).mean();x['EMA50']=c.ewm(span=50).mean();x['EMA200']=c.ewm(span=200).mean();x['MACD']=c.ewm(span=12).mean()-c.ewm(span=26).mean();x['MACDS']=x.MACD.ewm(span=9).mean();x['RET']=c.pct_change();x['VOL']=x.RET.rolling(20).std()*np.sqrt(252);x['VOLZ']=(v-v.rolling(20).mean())/v.rolling(20).std();x['ATR']=pd.concat([x.High-x.Low,(x.High-c.shift()).abs(),(x.Low-c.shift()).abs()],axis=1).max(axis=1).rolling(14).mean();return x
@st.cache_data(ttl=3300,show_spinner=False)
def reddit_access_token(client_id,client_secret,user_agent):
 if not client_id or not client_secret:return ''
 try:
  response=requests.post('https://www.reddit.com/api/v1/access_token',auth=(client_id,client_secret),data={'grant_type':'client_credentials'},headers={'User-Agent':user_agent},timeout=(3,8));response.raise_for_status();return str(response.json().get('access_token') or '')
 except (requests.RequestException,ValueError):return ''
@st.cache_data(ttl=300,show_spinner=False)
def news(company,symbol):
 out=[]
 try:
  out.extend(normalize_yahoo_news(yf.Ticker(symbol).get_news(count=30)))
 except Exception:
  pass
 url='https://news.google.com/rss/search?q='+urllib.parse.quote(company+' stock India')+'&hl=en-IN&gl=IN&ceid=IN:en'
 try:
  response=requests.get(url,timeout=(3,8));response.raise_for_status();feed=feedparser.parse(response.content)
  for e in feed.entries[:20]:
   published=e.get('published','')
   if e.get('published_parsed'):
    published=datetime(*e.published_parsed[:6],tzinfo=timezone.utc).isoformat()
   out.append({'title':e.get('title',''),'url':e.get('link',''),'source':e.get('source',{}).get('title','Google News'),'published':published})
 except Exception:
  pass
 bare_symbol=symbol.rsplit('.',1)[0]
 try:out.extend(fetch_gdelt_news(f'("{company}" OR "{bare_symbol}")',limit=20))
 except Exception:pass
 try:out.extend(fetch_bluesky_posts(f'{company} stock',limit=20))
 except Exception:pass
 reddit_id=os.getenv('REDDIT_CLIENT_ID','');reddit_secret=os.getenv('REDDIT_CLIENT_SECRET','');reddit_agent=os.getenv('REDDIT_USER_AGENT','indian-stock-ai/1.0')
 if reddit_id and reddit_secret:
  token=reddit_access_token(reddit_id,reddit_secret,reddit_agent)
  try:out.extend(fetch_reddit_posts(f'"{company}" OR "{bare_symbol}"',token,reddit_agent,limit=20))
  except Exception:pass
 articles=[];seen=set();now=datetime.now(timezone.utc).isoformat();c=conn()
 for item in out:
  title=item.get('title','').strip();key=title.casefold()
  if not title or key in seen:continue
  seen.add(key);score=AN.polarity_scores(item.get('sentiment_text',title))['compound'];item['score']=score;articles.append(item)
  c.execute('INSERT OR IGNORE INTO news_history(symbol,captured_at,published_at,source,headline,url,sentiment,relevance) VALUES(?,?,?,?,?,?,?,?)',(symbol,now,item.get('published',''),item.get('source','Financial News'),title,item.get('url',''),score,float(item.get('relevance',1))))
 c.commit();c.close();return articles
def sector_proxy(sec,suffix,period):
 frames=[]
 for t in SECTORS[sec].values():
  try:frames.append(hist(t+suffix,period).Close.rename(t))
  except:pass
 p=pd.concat(frames,axis=1).ffill();return (p/p.iloc[0]).mean(axis=1).rename('Close').to_frame()
def reconcile(symbol,d):
 closes={i.date().isoformat():float(v) for i,v in d.Close.items()};dates=sorted(closes);c=conn();rows=c.execute('SELECT * FROM predictions WHERE symbol=? AND actual_close IS NULL',(symbol,)).fetchall()
 for r in rows:
  future=[z for z in dates if z>r['prediction_date']]
  if len(future)<r['horizon']:continue
  td=future[r['horizon']-1];actual=closes[td];ar=actual/r['latest_close']-1;pdirection=np.sign(r['predicted_return']);adirection=np.sign(ar)
  c.execute('UPDATE predictions SET target_date=?,actual_close=?,actual_return=?,percentage_error=?,direction_correct=?,evaluated_at=? WHERE id=?',(td,actual,ar,abs(actual-r['predicted_close'])/actual*100,int(pdirection==adirection),datetime.now(timezone.utc).isoformat(),r['id']))
 c.commit();c.close()
def model(stock,market,sector,symbol,horizon,current_news,cost):
 x=stock.copy();m=market.Close.reindex(x.index).ffill();s=sector.Close.reindex(x.index).ffill();x['MKT']=m.pct_change();x['SEC']=s.pct_change();x['RELM']=x.RET-x.MKT;x['RELS']=x.RET-x.SEC
 x['RET5']=x.Close.pct_change(5);x['RET20']=x.Close.pct_change(20);x['MKT5']=m.pct_change(5);x['MKT20']=m.pct_change(20);x['MKT_VOL20']=x.MKT.rolling(20).std()*np.sqrt(252);x['SEC5']=s.pct_change(5);x['SEC20']=s.pct_change(20);x['SEC_VOL20']=x.SEC.rolling(20).std()*np.sqrt(252)
 c=conn();rows=c.execute('SELECT published_at,sentiment,relevance FROM news_history WHERE symbol=? AND published_at IS NOT NULL',(symbol,)).fetchall();c.close();x['NEWS']=point_in_time_news_feature(x.index,[dict(r) for r in rows]).to_numpy();x['Y']=x.Close.shift(-horizon)/x.Close-1;band=np.maximum(.004,x.RET.rolling(60).std()*np.sqrt(horizon)*.35);x['C']=np.select([x.Y>band,x.Y<-band],[2,0],default=1)
 F=['RSI','EMA20','EMA50','EMA200','MACD','MACDS','RET','RET5','RET20','VOL','VOLZ','ATR','MKT','MKT5','MKT20','MKT_VOL20','SEC','SEC5','SEC20','SEC_VOL20','RELM','RELS','NEWS'];z=x[F+['Y','C']].replace([np.inf,-np.inf],np.nan).dropna().iloc[:-horizon]
 if len(z)<260:raise ValueError('Choose 5y or 10y history; insufficient valid rows.')
 X,yr,yc=z[F],z.Y,z.C.astype(int);cv=TimeSeriesSplit(n_splits=5,gap=horizon);op=np.full(len(z),np.nan);oq=np.full((len(z),3),np.nan)
 for tr,te in cv.split(X):
  r1=RandomForestRegressor(n_estimators=250,max_depth=8,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X.iloc[tr],yr.iloc[tr]);r2=HistGradientBoostingRegressor(max_iter=180,max_depth=5,random_state=42).fit(X.iloc[tr],yr.iloc[tr]);cl=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=42).fit(X.iloc[tr],yc.iloc[tr]);op[te]=(r1.predict(X.iloc[te])+r2.predict(X.iloc[te]))/2;q=cl.predict_proba(X.iloc[te]);
  for j,k in enumerate(cl.classes_):oq[te,int(k)]=q[:,j]
 valid=~np.isnan(op);act=yr.to_numpy()[valid];pred=op[valid];classes=yc.to_numpy()[valid];probs=oq[valid];cal=LogisticRegression(max_iter=1000).fit(probs,classes);cp=cal.predict_proba(probs);pc=cp.argmax(1)
 metrics={'balanced_accuracy':balanced_accuracy_score(classes,pc)*100,'baseline_accuracy':balanced_accuracy_score(classes,np.ones_like(classes))*100,'mae':mean_absolute_error(act,pred)*100,'naive_mae':mean_absolute_error(act,np.zeros_like(act))*100,'buy_precision':precision_score(classes,pc,labels=[2],average='micro',zero_division=0)*100,'sell_precision':precision_score(classes,pc,labels=[0],average='micro',zero_division=0)*100,'brier':np.mean([brier_score_loss((classes==k).astype(int),cp[:,k]) for k in range(3)])}
 strategy=np.where(pc==2,act-cost/10000,np.where(pc==0,-act-cost/10000,0));metrics['cost_return']=(np.prod(1+strategy)-1)*100
 r1=RandomForestRegressor(n_estimators=400,max_depth=8,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X,yr);r2=HistGradientBoostingRegressor(max_iter=220,max_depth=5,random_state=42).fit(X,yr);cl=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=42).fit(X,yc);latest=x[F].dropna().iloc[[-1]].copy();latest['NEWS']=current_news;pr=float((r1.predict(latest)[0]+r2.predict(latest)[0])/2);raw=np.zeros((1,3));q=cl.predict_proba(latest)[0]
 for k,v in zip(cl.classes_,q):raw[0,int(k)]=v
 pp=cal.predict_proba(raw)[0];res=act-pred;lo,hi=np.quantile(res,[.1,.9]);close=float(stock.Close.iloc[-1]);version=f'{symbol}-h{horizon}-{stock.index[-1].date()}'
 return {'return':pr,'price':close*(1+pr),'low':close*(1+pr+lo),'high':close*(1+pr+hi),'signal':['SELL','HOLD','BUY'][pp.argmax()],'confidence':pp.max()*100,'bull':pp[2]*100,'bear':pp[0]*100,'metrics':metrics,'version':version,'features':latest.iloc[0].to_dict()}

st.title('📈 Bharat Market AI Production Research Platform');st.error('No model can guarantee returns. Use this only for research and paper trading, not as investment advice.')
with st.sidebar:
 exchange=st.selectbox('Exchange',['NSE','BSE']);suffix='.NS' if exchange=='NSE' else '.BO';sec=st.selectbox('Sector',list(SECTORS));company=st.selectbox('Company',list(SECTORS[sec]));symbol=SECTORS[sec][company]+suffix;hn=st.selectbox('Horizon',list(H));horizon=H[hn];period=st.selectbox('History',['2y','5y','10y'],index=1);cost=st.number_input('Round-trip cost (bps)',0,200,20,5);run=st.button('Train, evaluate & predict',type='primary',use_container_width=True)
try:
 raw=hist(symbol,period);stock=indicators(raw);market=hist('^NSEI',period);sector=sector_proxy(sec,suffix,period);articles=news(company,symbol);reconcile(symbol,raw);nscore=np.average([a['score'] for a in articles],weights=[a.get('relevance',1) for a in articles]) if articles else 0;mret=market.Close.pct_change().iloc[-1];sret=sector.Close.pct_change().iloc[-1];condition='Bullish' if mret>.003 else 'Bearish' if mret<-.003 else 'Neutral'
except Exception as e:st.error(str(e));st.stop()
tabs=st.tabs(['Prediction','Accuracy & Learning','News','Technical','Data Center','Operations'])
with tabs[0]:
 fig=go.Figure([go.Candlestick(x=stock.index,open=stock.Open,high=stock.High,low=stock.Low,close=stock.Close),go.Scatter(x=stock.index,y=stock.EMA20,name='EMA20'),go.Scatter(x=stock.index,y=stock.EMA50,name='EMA50')]);fig.update_layout(height=430,xaxis_rangeslider_visible=False);st.plotly_chart(fig,use_container_width=True)
 if run:
  try:
   r=model(stock,market,sector,symbol,horizon,nscore,cost);close=float(stock.Close.iloc[-1]);cols=st.columns(7);vals=[('Signal',r['signal']),('Confidence',f"{r['confidence']:.1f}%"),('Latest',f'₹{close:,.2f}'),('Forecast',f"₹{r['price']:,.2f}"),('Return',f"{r['return']*100:+.2f}%"),('Bullish',f"{r['bull']:.1f}%"),('Bearish',f"{r['bear']:.1f}%")]
   for c,(k,v) in zip(cols,vals):c.metric(k,v)
   st.write(f"80% empirical interval: **₹{r['low']:,.2f} to ₹{r['high']:,.2f}**")
   c=conn();c.execute('INSERT OR REPLACE INTO predictions(symbol,prediction_date,horizon,latest_close,predicted_close,predicted_return,signal,confidence,bullish,bearish,low,high,market_condition,market_return,sector_return,news_score,news_json,feature_json,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(symbol,stock.index[-1].date().isoformat(),horizon,close,r['price'],r['return'],r['signal'],r['confidence'],r['bull'],r['bear'],r['low'],r['high'],condition,mret,sret,nscore,json.dumps(articles),json.dumps(r['features']),r['version']));c.execute('INSERT INTO model_registry(symbol,horizon,version,trained_at,rows,metrics_json,active) VALUES(?,?,?,?,?,?,1)',(symbol,horizon,r['version'],datetime.now(timezone.utc).isoformat(),len(stock),json.dumps(r['metrics'])));c.commit();c.close()
   st.success('Prediction, inputs, news snapshot and model version permanently stored.')
   mc=st.columns(7);M=r['metrics'];ev=[('Balanced accuracy',M['balanced_accuracy']),('Hold baseline',M['baseline_accuracy']),('MAE',M['mae']),('Naive MAE',M['naive_mae']),('BUY precision',M['buy_precision']),('SELL precision',M['sell_precision']),('Cost return',M['cost_return'])]
   for c,(k,v) in zip(mc,ev):c.metric(k,f'{v:.2f}%')
   if M['balanced_accuracy']<=M['baseline_accuracy'] or M['mae']>=M['naive_mae'] or M['cost_return']<=0:st.warning('Production gate failed. Do not use this signal.')
  except Exception as e:st.error(str(e))
with tabs[1]:
 c=conn();rows=[dict(x) for x in c.execute('SELECT * FROM predictions WHERE symbol=? ORDER BY prediction_date DESC',(symbol,)).fetchall()];c.close()
 if rows:
  d=pd.DataFrame(rows);done=d.dropna(subset=['actual_close']);st.dataframe(d[['prediction_date','target_date','predicted_close','actual_close','percentage_error','signal','direction_correct','market_condition','model_version']],use_container_width=True,hide_index=True)
  if len(done):a=st.columns(4);a[0].metric('Live predictions',len(done));a[1].metric('Direction accuracy',f"{done.direction_correct.mean()*100:.1f}%");a[2].metric('Live MAE',f"{done.percentage_error.mean():.2f}%");a[3].metric('Versions',done.model_version.nunique())
 else:st.info('Run predictions. The app evaluates each one after its target date becomes available.')
with tabs[2]:
 st.metric('Current sentiment',f'{(nscore+1)*50:.1f}/100')
 st.caption('Feeds: Yahoo Finance, Google News RSS, GDELT, and public Bluesky search. Reddit search is enabled when Reddit API credentials are set in .env. Social posts are lower-weighted than news headlines.')
 for a in articles:st.markdown(f"[{'🟢' if a['score']>.05 else '🔴' if a['score']<-.05 else '🟡'} {a['title']}]({a['url']}) · {a['source']} · {a['score']:+.2f}")
with tabs[3]:
 cols=st.columns(8)
 for c,k in zip(cols,['RSI','MACD','MACDS','EMA20','EMA50','EMA200','ATR','VOL']):c.metric(k,f"{stock[k].iloc[-1]:.2f}")
with tabs[4]:
 st.subheader('Upload point-in-time historical news');f=st.file_uploader('News CSV',type='csv')
 if f:
  d=pd.read_csv(f);req={'published_at','symbol','headline','source','sentiment'}
  if not req<=set(d.columns):st.error('Missing columns: '+str(req-set(d.columns)))
  else:
   d.published_at=pd.to_datetime(d.published_at,utc=True,errors='coerce');d.sentiment=pd.to_numeric(d.sentiment,errors='coerce').clip(-1,1);d['relevance']=pd.to_numeric(d.get('relevance',1),errors='coerce').fillna(1).clip(0,1);d['url']=d.get('url','');st.dataframe(d.head())
   if st.button('Import validated news'):
    c=conn();count=0
    for _,z in d.dropna(subset=['published_at','symbol','headline','sentiment']).iterrows():count+=c.execute('INSERT OR IGNORE INTO news_history(symbol,captured_at,published_at,source,headline,url,sentiment,relevance) VALUES(?,?,?,?,?,?,?,?)',(z.symbol,z.published_at.isoformat(),z.published_at.isoformat(),z.source,z.headline,z.url,float(z.sentiment),float(z.relevance))).rowcount
    c.commit();c.close();st.success(f'Imported {count} articles')
 st.download_button('News template',open('sample_news.csv','rb').read(),'sample_news.csv')
with tabs[5]:
 st.markdown('''**Daily learning process**
1. Run after market close to capture finalized prices and timestamped news.
2. Preserve `stock_ai.db`; it contains the feedback loop.
3. Re-run after the target date to reconcile forecasts with actual closes.
4. Add licensed historical news through Data Center.
5. Promote a challenger only when it beats direction and price baselines, stays calibrated, and remains profitable after costs.

For real multi-user production, migrate SQLite to managed PostgreSQL, run ingestion/training as scheduled workers, keep immutable raw data in object storage, manage models in a registry, and protect the app with SSO, WAF, backups and monitoring.''')
