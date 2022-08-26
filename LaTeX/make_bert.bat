@echo off
for /r %%i in (Main_*BERT*.tex) do texify -cp %%i
