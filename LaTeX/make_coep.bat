@echo off
for /r %%i in (Main_*CoEP*.tex) do texify -cp %%i
