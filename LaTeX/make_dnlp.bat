@echo off
for /r %%i in (Main_*_DeepNLP*.tex) do texify -cp %%i
