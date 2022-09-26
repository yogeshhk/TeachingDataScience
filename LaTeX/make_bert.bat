@echo off
for /r %%i in (Main*BERT*.tex) do texify -cp %%i
