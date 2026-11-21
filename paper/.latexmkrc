# latexmk config — run `latexmk` in this folder
$pdf_mode = 1;                 # pdflatex
$out_dir  = 'build';
$aux_dir  = 'build';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
@default_files = ('main.tex');
