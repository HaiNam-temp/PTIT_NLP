import nbformat, sys
from nbclient import NotebookClient
nb = nbformat.read("BTL1_PTIT_SPAM_HAM_Kaggle.ipynb", as_version=4)
try:
    NotebookClient(nb, timeout=3000, kernel_name="python3").execute()
    print("DONE")
except Exception as e:
    print("FAILED", str(e)[-3000:])
nbformat.write(nb, "scratch/executed.ipynb")
