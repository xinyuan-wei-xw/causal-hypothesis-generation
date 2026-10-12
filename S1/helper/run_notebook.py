"""Execute a notebook in Python; always save outputs and a timed full HTML export.

Usage: python S1/helper/run_notebook.py S1/S1_02_causal_forest_policy_trees.ipynb
"""
from pathlib import Path
from datetime import datetime
import sys,time,traceback,base64
import nbformat
from nbconvert import HTMLExporter
from IPython.utils.capture import capture_output

def run(path):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.core.pylabtools import print_figure
    from matplotlib.figure import Figure
    shell=InteractiveShell.instance()
    shell.display_formatter.formatters['image/png'].for_type(Figure,lambda fig: print_figure(fig,fmt='png'))
    path=Path(path).resolve();nb=nbformat.read(path,4);namespace={};start=time.perf_counter();timings=[];failure=None
    for cell in nb.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    try:
        for i,cell in enumerate(nb.cells):
            if cell.cell_type!='code':continue
            began=time.perf_counter();print(f'{path.name}: cell {i+1}',flush=True)
            with capture_output() as captured:
                try:exec(compile(cell.source,str(path),'exec'),namespace)
                except BaseException as exc:failure=exc;traceback.print_exc()
            cell.execution_count=len(timings)+1
            for stream,text in [('stdout',captured.stdout),('stderr',captured.stderr)]:
                if text:cell.outputs.append(nbformat.v4.new_output('stream',name=stream,text=text))
            for output in captured.outputs:
                data={mime:base64.b64encode(value).decode() if isinstance(value,bytes) else value for mime,value in output.data.items()}
                cell.outputs.append(nbformat.v4.new_output('display_data',data=data,metadata=output.metadata))
            timings.append({'cell':i+1,'seconds':time.perf_counter()-began})
            print(f'  {timings[-1]["seconds"]:.2f}s',flush=True)
            if failure:
                cell.outputs.append(nbformat.v4.new_output('error',ename=type(failure).__name__,evalue=str(failure),traceback=[str(failure)]));break
    except BaseException as exc:
        failure=exc
        raise
    finally:
        elapsed=time.perf_counter()-start
        nb.metadata['execution_review']={'elapsed_seconds':elapsed,'status':'failed' if failure else 'completed','cell_timings':timings}
        nbformat.write(nb,path)
        review=nbformat.from_dict(nb)
        timing_text=", ".join(str(r["cell"])+": "+format(r["seconds"],".2f") for r in timings)
        review.cells.insert(0,nbformat.v4.new_markdown_cell(f"Execution status: {nb.metadata.execution_review.status}. Total runtime: {elapsed:.2f} seconds.\n\n"+datetime.now().astimezone().isoformat()+"\n\nCell runtimes: "+timing_text))
        folder=path.parent/'reports';folder.mkdir(exist_ok=True)
        export=folder/f'{path.stem}_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}_full.html'
        body,_=HTMLExporter().from_notebook_node(review);export.write_text(body)
        print(f'{nb.metadata.execution_review.status}: {elapsed:.2f}s; {export}',flush=True)
    if failure:raise failure

if __name__=='__main__':run(sys.argv[1])
