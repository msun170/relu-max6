# Building the paper

The paper is a single self-contained LaTeX file, `max6.tex` (article class + amsmath/amsthm/hyperref).

## Easiest: Overleaf
Upload `max6.tex` to a new Overleaf project and compile. No local setup.

## Local, no TeX install: tectonic (recommended)
[Tectonic](https://tectonic-typesetting.github.io/) is a single self-contained binary that fetches the TeX
support files it needs on first run and caches them.

```
tectonic max6.tex          # produces max6.pdf
```

A Windows build was placed at `C:\Users\nuswe\bin\tectonic.exe` on this machine, so:

```
C:\Users\nuswe\bin\tectonic.exe max6.tex
```

Note: on the first run tectonic downloads its TeX bundle over the network. If HTTPS is intercepted by a
local security product (this machine uses Norton, which MITMs TLS), point tectonic at a CA bundle that
includes the interception root, e.g.:

```
set SSL_CERT_FILE=<certifi bundle concatenated with C:\ProgramData\Norton\Antivirus\wscert.pem>
```

After the first successful run the bundle is cached under `%LOCALAPPDATA%\TectonicProject`, and later builds
work offline.

## Local, full TeX distribution
With MiKTeX or TeX Live installed:

```
pdflatex max6.tex && pdflatex max6.tex     # twice, to resolve cross-references
```
