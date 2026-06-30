# Register the nvidia pip-wheel DLL directories so cupy can load cusolver (eigh/svd/qr) on Windows.
# Import this BEFORE `import cupy`. Fixes "ImportError: DLL load failed while importing cusolver".
import os, glob, site
for b in site.getsitepackages() + [site.getusersitepackages()]:
    for d in glob.glob(os.path.join(b, "nvidia", "*", "bin")):
        try: os.add_dll_directory(d)
        except Exception: pass
