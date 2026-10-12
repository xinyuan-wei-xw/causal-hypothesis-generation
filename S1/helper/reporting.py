"""HTML review reports with fresh cell output and execution timing."""
from pathlib import Path
from datetime import datetime
import time,html,base64
import pandas as pd
from nbconvert.filters.markdown import markdown2html

class Review:
    def __init__(self,folder,name):
        self.started=time.perf_counter();self.last=self.started
        self.name=name;self.folder=Path(folder);self.parts=[];self.timings=[]
    def note(self,text):
        self.parts.append(markdown2html(text))
    def table(self,title,frame):
        self.parts.append('<h3>'+html.escape(title)+'</h3>'+frame.to_html(index=False,float_format=lambda x:f'{x:.4f}'))
    def figure(self,title,path):
        self.parts.append('<h3>'+html.escape(title)+'</h3><img alt="'+html.escape(title)+'" src="data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'">')
    def mark(self,stage):
        now=time.perf_counter();self.timings.append({'stage':stage,'seconds':now-self.last});self.last=now
    def export(self):
        elapsed=time.perf_counter()-self.started
        self.folder.mkdir(parents=True,exist_ok=True)
        path=self.folder/f'{self.name}_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}.html'
        timing=pd.DataFrame(self.timings+[{'stage':'Total elapsed','seconds':elapsed}])
        content='<h1>'+html.escape(self.name)+'</h1><p>Executed '+datetime.now().astimezone().isoformat()+f'; elapsed {elapsed:.2f} seconds.</p>'+timing.to_html(index=False)+''.join(self.parts)
        path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(self.name)+'</title><style>body{font:16px/1.55 system-ui;margin:30px;max-width:1500px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}pre{overflow:auto}</style><body>'+content+'</body></html>')
        timing.to_csv(path.with_suffix('.timing.csv'),index=False)
        return path
