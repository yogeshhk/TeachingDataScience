@echo off
for /r %%i in (Main_*Tensorflow_*.tex) do texify -cp %%i
