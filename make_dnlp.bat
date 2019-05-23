@echo off
for /r %%i in (Main_Workshop_DeepNLP*.tex) do texify -cp %%i
